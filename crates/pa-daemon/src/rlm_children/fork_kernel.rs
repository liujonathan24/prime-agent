//! The kernel half of `rlm.spawn(fork_from=...)`: copy the source
//! session's kernel snapshot (`kernel-state.dill` + its manifest) into the
//! forked child's artifact dir before the child's worker exists, so the
//! existing resume prewarm revives the source's Python namespace.
//!
//! A settled child writes its snapshot asynchronously: the debounced
//! auto-snapshot after each successful cell, and the flush when its worker
//! parks (`maybe_release_settled_child_kernel`). A factory `"self"` fork
//! spawns right after the settle verdict, which can land before that
//! flush. The copy therefore waits (bounded) for a snapshot whose manifest
//! is at least as new as the source's last Python tool result: once that
//! holds, no cell ran after the snapshot, so it is the source's latest
//! kernel. A source without a Python tool result has nothing to wait for.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use pa_core::kernel::{manifest_path_in, snapshot_path_in};

/// Upper bound on the wait for the source's fresh snapshot. A source whose
/// kernel is never released (a root or client-attached session) never
/// writes one; the fork then starts without a kernel instead of waiting
/// forever or copying a stale namespace.
const FRESH_SNAPSHOT_WAIT: Duration = Duration::from_secs(30);
const FRESH_SNAPSHOT_POLL: Duration = Duration::from_millis(100);

/// The artifact dir a session file resolves to, by the same rule as
/// `SessionManager::get_session_artifact_dir`:
/// `dirname(session dir)/session-artifacts/<header id>`.
fn artifact_dir_of(session_file: &Path) -> Option<PathBuf> {
    artifact_dir_for(session_file, &read_session(session_file).header_id?)
}

fn artifact_dir_for(session_file: &Path, id: &str) -> Option<PathBuf> {
    let session_dir = session_file.parent()?;
    Some(session_dir.parent()?.join("session-artifacts").join(id))
}

#[derive(Default)]
struct SessionScan {
    header_id: Option<String>,
    /// Epoch ms of the newest Python tool result, if any.
    last_python_result_ms: Option<u64>,
}

fn read_session(session_file: &Path) -> SessionScan {
    let mut scan = SessionScan::default();
    let Ok(text) = std::fs::read_to_string(session_file) else {
        return scan;
    };
    for line in text.lines() {
        // A torn tail row is skipped, like the session loader does.
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match row.get("type").and_then(serde_json::Value::as_str) {
            Some("session") if scan.header_id.is_none() => {
                scan.header_id = row
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
            }
            Some("message") => {
                let Some(message) = row.get("message") else {
                    continue;
                };
                let python_result = message.get("role").and_then(serde_json::Value::as_str)
                    == Some("toolResult")
                    && message
                        .get("toolName")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(pa_types::ai::is_python_tool_name);
                if let Some(at) = message
                    .get("timestamp")
                    .and_then(serde_json::Value::as_u64)
                    .filter(|_| python_result)
                {
                    scan.last_python_result_ms =
                        Some(scan.last_python_result_ms.map_or(at, |last| last.max(at)));
                }
            }
            _ => {}
        }
    }
    scan
}

fn modified_ms(path: &Path) -> Option<u64> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    u64::try_from(modified.duration_since(UNIX_EPOCH).ok()?.as_millis()).ok()
}

/// Copy the source's snapshot into the child's artifact dir when it is
/// fresh. Returns `Ok(true)` when a snapshot was copied, `Ok(false)` when
/// the source has none (or none turned fresh within the bound).
pub(super) async fn copy_fork_kernel(
    source_file: &Path,
    child_file: &Path,
) -> anyhow::Result<bool> {
    let source_file = source_file.to_path_buf();
    let child_file = child_file.to_path_buf();
    let deadline = Instant::now() + FRESH_SNAPSHOT_WAIT;
    loop {
        let (source, child) = (source_file.clone(), child_file.clone());
        let attempt = tokio::task::spawn_blocking(move || try_copy(&source, &child)).await??;
        match attempt {
            Attempt::Copied => return Ok(true),
            Attempt::NoSnapshot => return Ok(false),
            Attempt::Stale if Instant::now() >= deadline => {
                eprintln!(
                    "pa-daemon: fork_from {}: no kernel snapshot newer than the source's last \
                     python result within {}s; the child starts without the source's kernel",
                    source_file.display(),
                    FRESH_SNAPSHOT_WAIT.as_secs()
                );
                return Ok(false);
            }
            Attempt::Stale => tokio::time::sleep(FRESH_SNAPSHOT_POLL).await,
        }
    }
}

enum Attempt {
    Copied,
    /// The source never ran Python and has no snapshot.
    NoSnapshot,
    /// A Python result is newer than the snapshot (or no snapshot yet).
    Stale,
}

fn try_copy(source_file: &Path, child_file: &Path) -> anyhow::Result<Attempt> {
    let scan = read_session(source_file);
    let Some(source_dir) = scan
        .header_id
        .as_deref()
        .and_then(|id| artifact_dir_for(source_file, id))
    else {
        return Ok(Attempt::NoSnapshot);
    };
    let (snapshot, manifest) = (snapshot_path_in(&source_dir), manifest_path_in(&source_dir));
    let manifest_at = modified_ms(&manifest).filter(|_| snapshot.is_file());
    match (manifest_at, scan.last_python_result_ms) {
        (None, None) => return Ok(Attempt::NoSnapshot),
        (None, Some(_)) => return Ok(Attempt::Stale),
        (Some(at), Some(last)) if at < last => return Ok(Attempt::Stale),
        (Some(_), _) => {}
    }
    let child_dir = artifact_dir_of(child_file)
        .ok_or_else(|| anyhow::anyhow!("forked session {} has no header", child_file.display()))?;
    std::fs::create_dir_all(&child_dir)?;
    // The writer replaces the payload before the manifest: a manifest that
    // changed while the payload was copied means a newer snapshot landed
    // mid-copy, so the pair is read again.
    std::fs::copy(&snapshot, snapshot_path_in(&child_dir))?;
    std::fs::copy(&manifest, manifest_path_in(&child_dir))?;
    if modified_ms(&manifest) != manifest_at {
        return Ok(Attempt::Stale);
    }
    Ok(Attempt::Copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(dir: &Path, id: &str, python_result_at: Option<u64>) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let file = dir.join(format!("{id}.jsonl"));
        let mut rows = vec![serde_json::json!({"type": "session", "version": 3, "id": id})];
        if let Some(at) = python_result_at {
            rows.push(serde_json::json!({
                "type": "message",
                "id": "r1",
                "message": {"role": "toolResult", "toolName": "python_repl", "timestamp": at}
            }));
        }
        let lines: Vec<String> = rows.iter().map(ToString::to_string).collect();
        std::fs::write(&file, lines.join("\n") + "\n").unwrap();
        file
    }

    fn snapshot(artifact_dir: &Path) {
        std::fs::create_dir_all(artifact_dir).unwrap();
        std::fs::write(snapshot_path_in(artifact_dir), b"payload").unwrap();
        std::fs::write(manifest_path_in(artifact_dir), b"{}").unwrap();
    }

    fn now_ms() -> u64 {
        u64::try_from(
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap()
    }

    #[test]
    fn the_artifact_dir_follows_the_session_manager_rule() {
        let tmp = tempfile::tempdir().unwrap();
        let file = session(&tmp.path().join("sub-1"), "s-1", None);
        assert_eq!(
            artifact_dir_of(&file),
            Some(tmp.path().join("session-artifacts").join("s-1"))
        );
    }

    #[tokio::test]
    async fn a_fresh_snapshot_is_copied_into_the_childs_artifact_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let source = session(&tmp.path().join("src"), "s-1", Some(now_ms() - 5_000));
        snapshot(&tmp.path().join("session-artifacts").join("s-1"));
        let child = session(&tmp.path().join("kids").join("sub-2"), "s-2", None);
        assert!(copy_fork_kernel(&source, &child).await.unwrap());
        let child_dir = tmp
            .path()
            .join("kids")
            .join("session-artifacts")
            .join("s-2");
        assert_eq!(
            std::fs::read(snapshot_path_in(&child_dir)).unwrap(),
            b"payload"
        );
        assert!(manifest_path_in(&child_dir).is_file());
    }

    #[tokio::test]
    async fn a_source_that_never_ran_python_forks_without_a_kernel_at_once() {
        let tmp = tempfile::tempdir().unwrap();
        let source = session(&tmp.path().join("src"), "s-1", None);
        let child = session(&tmp.path().join("kids").join("sub-2"), "s-2", None);
        let started = Instant::now();
        assert!(!copy_fork_kernel(&source, &child).await.unwrap());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn the_copy_waits_for_the_snapshot_a_later_flush_writes() {
        let tmp = tempfile::tempdir().unwrap();
        // The source's last cell ran after the only snapshot it has.
        let artifact_dir = tmp.path().join("session-artifacts").join("s-1");
        snapshot(&artifact_dir);
        let source = session(&tmp.path().join("src"), "s-1", Some(now_ms() + 1_000));
        let child = session(&tmp.path().join("kids").join("sub-2"), "s-2", None);
        let flush = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1_500)).await;
            std::fs::write(snapshot_path_in(&artifact_dir), b"fresh").unwrap();
            std::fs::write(manifest_path_in(&artifact_dir), b"{}").unwrap();
        });
        assert!(copy_fork_kernel(&source, &child).await.unwrap());
        flush.await.unwrap();
        let child_dir = tmp
            .path()
            .join("kids")
            .join("session-artifacts")
            .join("s-2");
        assert_eq!(
            std::fs::read(snapshot_path_in(&child_dir)).unwrap(),
            b"fresh"
        );
    }
}
