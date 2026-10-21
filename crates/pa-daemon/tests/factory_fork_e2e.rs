//! Factory fork e2e: a library machine run from a faux parent's kernel cell
//! drives `fork_from` and `mcp_servers` through the whole product stack --
//! `rlm.factory.run` -> factory executor -> `rlm.spawn` kwargs -> the daemon's
//! child spawn (session copy + kernel snapshot copy + session-scoped MCP
//! install) -> the child's kernel. Every model turn is scripted
//! (`"engine": "faux"`), so the cells run in real kernels with no provider.
//! A third test checks that a spawn made while the parent is idle, as a
//! factory's background task does, gets its task prompt at once.
#![allow(clippy::large_futures)]
// Fn length is a style gate, not correctness.
#![allow(clippy::too_many_lines)]
#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

struct Daemon {
    child: Child,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The kernel Python with prime-agent-runtime installed; skipped (with a note)
/// without a live install.
fn kernel_python() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("PA_E2E_KERNEL_PYTHON") {
        let explicit = PathBuf::from(explicit);
        assert!(
            explicit.exists(),
            "PA_E2E_KERNEL_PYTHON {} not found",
            explicit.display()
        );
        return Some(explicit);
    }
    let candidate = PathBuf::from(std::env::var("HOME").map_or_else(
        |_| "/home/ubuntu/.prime/agent/kernel-venv/bin/python".to_string(),
        |home| format!("{home}/.prime/agent/kernel-venv/bin/python"),
    ));
    if candidate.exists() {
        return Some(candidate);
    }
    eprintln!(
        "kernel python {} not found; skipping live factory fork e2e",
        candidate.display()
    );
    None
}

#[allow(clippy::zombie_processes)]
fn spawn_supervisor(socket: &Path, agent_dir: &Path, kernel_python: &Path) -> Daemon {
    let binary = env!("CARGO_BIN_EXE_pa-daemon");
    let child = Command::new(binary)
        .arg("supervisor")
        .arg("--socket")
        .arg(socket)
        .arg("--agent-dir")
        .arg(agent_dir)
        .env("PRIME_AGENT_KERNEL_PYTHON", kernel_python)
        // Hermetic agent dir: settings, the user machine library and every
        // fallback resolve inside the test's temp dir.
        .env("PRIME_AGENT_CODING_AGENT_DIR", agent_dir)
        .env_remove("PRIME_AGENT_MACHINES_DIR")
        .env_remove("PRIME_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        // A supervisor killed at teardown must not leak its session workers into later
        // test binaries: the supervisor-lost exit runs here.
        .env(
            pa_daemon::worker::WORKER_SUPERVISOR_LOST_EXIT_MS_ENV,
            "15000",
        )
        .spawn()
        .expect("spawn pa-daemon supervisor");
    Daemon { child }
}

fn wait_socket_ready(socket: &Path) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if UnixStream::connect(socket).is_ok() {
            return;
        }
        assert!(Instant::now() < deadline, "supervisor socket never came up");
        std::thread::sleep(Duration::from_millis(20));
    }
}

struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Client {
    fn connect(socket: &Path) -> (Self, Value) {
        let stream = UnixStream::connect(socket).expect("connect supervisor");
        let write_stream = stream.try_clone().expect("clone socket");
        let mut client = Client {
            reader: BufReader::new(stream),
            writer: write_stream,
        };
        let hello = client.read_line();
        (client, hello)
    }

    fn send_command(&mut self, id: &str, command: &Value) {
        let mut line = serde_json::to_string(&json!({
            "type": "command",
            "id": id,
            "protocol": { "name": "prime-agent.daemon", "version": 7 },
            "command": command,
        }))
        .expect("serialize command");
        line.push('\n');
        self.writer.write_all(line.as_bytes()).expect("send");
        self.writer.flush().expect("flush");
    }

    fn read_line(&mut self) -> Value {
        let mut line = String::new();
        let deadline = Instant::now() + Duration::from_mins(2);
        self.reader
            .get_mut()
            .set_read_timeout(Some(Duration::from_millis(100)))
            .expect("set timeout");
        loop {
            match self.reader.read_line(&mut line) {
                Ok(0) => panic!("supervisor closed the connection"),
                Ok(_) if line.trim().is_empty() => line.clear(),
                Ok(_) => return serde_json::from_str(line.trim()).expect("parse response line"),
                Err(error) => {
                    assert!(
                        Instant::now() < deadline,
                        "timed out waiting for supervisor line: {error}"
                    );
                }
            }
        }
    }

    fn read_response(&mut self, id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_mins(4);
        loop {
            assert!(Instant::now() < deadline, "no response for id {id}");
            let line = self.read_line();
            if line.get("id").and_then(Value::as_str) == Some(id) {
                return line;
            }
        }
    }
}

/// Every JSONL row of a session file.
fn session_rows(path: &str) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read session file {path}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("session row json"))
        .collect()
}

/// The text of the LAST tool result in a session file: the child's own probe
/// cell (a forked child also carries its source's earlier results).
fn last_tool_result(path: &str) -> Value {
    let text = session_rows(path)
        .iter()
        .rev()
        .find_map(|row| {
            let message = row.get("message")?;
            (message.get("role").and_then(Value::as_str) == Some("toolResult")).then(|| {
                message["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<String>()
            })
        })
        .unwrap_or_else(|| panic!("no tool result in {path}"));
    let probe = text
        .lines()
        .find(|line| line.starts_with("{\"probe\""))
        .unwrap_or_else(|| panic!("no probe line in the last tool result of {path}: {text}"));
    serde_json::from_str(probe).expect("probe json")
}

/// The machine: `seed` hands a server list to `loop`, which installs it on
/// each child (`mcp_servers`) and continues its own previous round
/// (`fork_from`, `"self"` or the kernel opt-out) on the second entry.
const MACHINE: &str = r#"---
name: forkprobe
description: Two states that exercise fork_from and mcp_servers.
version: 1
author: e2e
---

# forkprobe

```machine-spec
{
  "run": {"max_parallel": 1, "max_transitions": 8},
  "states": [
    {"id": "seed", "entry": true,
     "outputs": [{"name": "servers", "type": "json"}],
     "subagent": {"name": "seed", "prompt": "Hand the server list on."}},
    {"id": "loop", "max_entries": 2,
     "inputs": [{"name": "servers", "type": "json", "from": "seed.servers", "hidden": true}],
     "outputs": [{"name": "note", "type": "json"}],
     "subagent": {"name": "loop", "prompt": "Probe the echo server.",
                  "fork_from": FORK_FROM,
                  "mcp_servers": {"input": "servers", "path": "list"}}}
  ],
  "transitions": [
    {"from": "seed", "to": "loop"},
    {"from": "loop", "to": "loop"}
  ]
}
```
"#;

#[test]
fn a_machine_forks_its_own_rounds_and_installs_session_mcp_servers() {
    // Round 2 spawns right after round 1 settles, before or after round 1's
    // worker flushes its kernel snapshot: the copy must wait for that flush.
    run_forkprobe(&json!("self"), 42);
}

#[test]
fn a_fork_without_the_kernel_starts_a_fresh_namespace() {
    run_forkprobe(&json!({ "self": true, "kernel": false }), 41);
}

/// Run the machine with `fork_from` and check the rounds; `second_memo` is
/// what round 2's probe counts: 42 when round 1's namespace came along.
fn run_forkprobe(fork_from: &Value, second_memo: u64) {
    let Some(kernel_python) = kernel_python() else {
        return;
    };
    let dir = tempfile::TempDir::new().expect("temp dir");
    let agent_dir = dir.path().join("agent");
    let sessions_dir = agent_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
    // The factory is opt-in; the machine sits in the user library.
    std::fs::write(
        agent_dir.join("settings.json"),
        json!({ "factory": { "enabled": true } }).to_string(),
    )
    .expect("write settings.json");
    let machine_dir = agent_dir.join("machines").join("forkprobe");
    std::fs::create_dir_all(&machine_dir).expect("machine dir");
    std::fs::write(
        machine_dir.join("MACHINE.md"),
        MACHINE.replace("FORK_FROM", &fork_from.to_string()),
    )
    .expect("write MACHINE.md");

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mcp_echo_server.py");
    assert!(fixture.exists(), "fixture missing: {}", fixture.display());
    // Session-scoped, NOT in settings: only a child it is installed on sees it.
    let server = json!({
        "type": "stdio",
        "name": "fixture-echo",
        "command": "python3",
        "args": [fixture.to_string_lossy()],
        "cwd": dir.path().to_string_lossy(),
    });

    // Every child runs the same script: count `memo` up from what its kernel
    // holds (40 in a fresh namespace), probe the echo server, print the
    // verdict, then answer one fenced block that binds both states' outputs.
    let probe = [
        "import json",
        "from rlm import mcp",
        "memo = globals().get(\"memo\", 40) + 1",
        "verdict = {\"probe\": True, \"memo\": memo}",
        "try:",
        "    verdict[\"tools\"] = [tool[\"name\"] for tool in await mcp.list_tools(\"fixture-echo\")]",
        "except Exception as error:",
        "    verdict[\"error\"] = str(error)",
        "print(json.dumps(verdict))",
    ]
    .join("\n");
    let answer = format!(
        "```json\n{}\n```",
        json!({ "servers": { "list": [server] }, "note": { "ok": true } })
    );
    let child_script = dir.path().join("child.json");
    std::fs::write(
        &child_script,
        json!({
            "engine": "faux",
            "responses": [
                { "content": [
                    { "type": "toolCall", "name": "python_repl", "arguments": { "code": probe } },
                ] },
                { "text": answer },
            ],
        })
        .to_string(),
    )
    .expect("write child script");

    // The parent starts the machine and ends its turn, as an interactive user
    // would, and answers every child-exit wake with a short reply: a child's
    // task prompt waits for the parent turn that spawned it, so a turn held
    // open on the run would delay each child.
    let error_receipt = dir.path().join("start.error");
    let start = [
        "import traceback",
        "try:",
        "    run = await rlm.factory.run(\"forkprobe\")",
        "except Exception:",
        "    open(ERROR_RECEIPT, \"w\").write(traceback.format_exc())",
        "    raise",
    ]
    .join("\n")
    .replace(
        "ERROR_RECEIPT",
        &format!("{:?}", error_receipt.display().to_string()),
    );
    let script = dir.path().join("parent.json");
    std::fs::write(
        &script,
        json!({
            "engine": "faux",
            "repeatLastResponse": true,
            "responses": [
                { "content": [
                    { "type": "toolCall", "name": "python_repl", "arguments": { "code": start } },
                ] },
                { "text": "noted" },
            ],
        })
        .to_string(),
    )
    .expect("write parent script");

    let socket = dir.path().join("factory.sock");
    let _daemon = spawn_supervisor(&socket, &agent_dir, &kernel_python);
    wait_socket_ready(&socket);
    let (mut client, hello) = Client::connect(&socket);
    assert_eq!(hello["type"], "daemon_hello");

    client.send_command(
        "c1",
        &json!({
            "type": "create",
            "config": {
                "cwd": dir.path().to_string_lossy(),
                "sessionDir": sessions_dir.to_string_lossy(),
                "script": script.to_string_lossy(),
                "childScript": child_script.to_string_lossy(),
            },
        }),
    );
    let created = client.read_response("c1");
    assert_eq!(created["success"], true, "create failed: {created}");
    let session_id = created["data"]["activeSessionId"]
        .as_str()
        .or_else(|| created["data"]["id"].as_str())
        .expect("active session id")
        .to_string();
    client.send_command(
        "p1",
        &json!({ "type": "prompt", "activeSessionId": session_id, "message": "start forkprobe" }),
    );
    assert_eq!(client.read_response("p1")["success"], true, "prompt failed");

    // The spawn ledger names each child's session file in spawn order; the
    // run is over once loop's second round has answered.
    let ledger_dir = agent_dir.join("rlm-ledger");
    let deadline = Instant::now() + Duration::from_secs(150);
    let children = loop {
        if let Ok(error) = std::fs::read_to_string(&error_receipt) {
            panic!("the machine failed to start in the parent's cell: {error}");
        }
        let children: Vec<(String, String)> = std::fs::read_dir(&ledger_dir)
            .into_iter()
            .flatten()
            .flatten()
            .flat_map(|entry| session_rows(&entry.path().to_string_lossy()))
            .filter(|row| row["op"] == "spawn")
            .filter_map(|row| {
                Some((
                    row["name"].as_str()?.to_string(),
                    row["child"].as_str()?.to_string(),
                ))
            })
            .collect();
        // Round 2's file begins with round 1's copied kickoff and answer:
        // round 2 has answered once an answer follows its own (second) kickoff.
        let answered = children.len() == 3 && {
            let rows = session_rows(&children[2].1);
            let kickoffs: Vec<usize> = rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row["customType"] == "agent_message")
                .map(|(index, _)| index)
                .collect();
            kickoffs.len() == 2
                && rows[kickoffs[1]..].iter().any(|row| {
                    row["message"]["role"] == "assistant"
                        && row["message"]["content"].to_string().contains("```json")
                })
        };
        if answered {
            break children;
        }
        assert!(
            Instant::now() < deadline,
            "the machine never reached loop's second answer; spawned: {children:?}"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    let [(seed_name, seed), (first_name, first), (second_name, second)] =
        <[(String, String); 3]>::try_from(children).expect("three children");
    assert!(
        seed_name.ends_with("-seed") && first_name.ends_with("-loop"),
        "spawn order: {seed_name}, {first_name}, {second_name}"
    );

    // fork_from "self": the first round starts fresh (its header names the
    // spawning parent, like seed's); the second is its own session, copied
    // from the first round's, and keeps every entry of it.
    let header = |path: &str| session_rows(path)[0].clone();
    assert_eq!(
        header(&first)["parentSession"],
        header(&seed)["parentSession"]
    );
    assert_eq!(header(&second)["parentSession"], first.as_str());
    assert_ne!(header(&second)["id"], header(&first)["id"]);
    let ids = |path: &str| -> Vec<String> {
        session_rows(path)
            .iter()
            .skip(1)
            .filter_map(|row| row["id"].as_str().map(str::to_string))
            .collect()
    };
    let (first_ids, second_ids) = (ids(&first), ids(&second));
    assert!(
        second_ids.len() > first_ids.len() && second_ids[..first_ids.len()] == first_ids[..],
        "round 2 must begin with round 1's entries:\n{first_ids:?}\n{second_ids:?}"
    );

    // mcp_servers: only the children it was installed on see the server.
    // The kernel: round 2 counts on from round 1's `memo` only when the fork
    // copied round 1's kernel snapshot.
    assert_eq!(
        last_tool_result(&seed),
        json!({ "probe": true, "memo": 41, "error": "\"MCP server 'fixture-echo' is not declared in user settings\"" }),
    );
    assert_eq!(
        last_tool_result(&first),
        json!({ "probe": true, "memo": 41, "tools": ["echo"] }),
    );
    assert_eq!(
        last_tool_result(&second),
        json!({ "probe": true, "memo": second_memo, "tools": ["echo"] }),
    );
}

/// A factory spawns from a background task in the parent's kernel, often
/// after the parent has ended its turn. Such a spawn has no parent turn to
/// wait for: its task prompt must reach the child at once, not when the
/// parent's next turn ends or the 60 s bound runs out. The parent's cell
/// schedules one `rlm.spawn` two seconds out and the parent ends its turn,
/// so the spawn lands while the parent is idle and nothing else wakes it.
#[test]
fn a_spawn_while_the_parent_is_idle_gets_its_task_at_once() {
    let Some(kernel_python) = kernel_python() else {
        return;
    };
    let dir = tempfile::TempDir::new().expect("temp dir");
    let agent_dir = dir.path().join("agent");
    let sessions_dir = agent_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
    let error_receipt = dir.path().join("spawn.error");
    let later = [
        "import asyncio, traceback",
        "async def spawn_later():",
        "    await asyncio.sleep(2)",
        "    try:",
        "        await rlm.spawn(\"idle task\", name=\"late\")",
        "    except Exception:",
        "        open(ERROR_RECEIPT, \"w\").write(traceback.format_exc())",
        "spawn_task = asyncio.get_running_loop().create_task(spawn_later())",
    ]
    .join("\n")
    .replace(
        "ERROR_RECEIPT",
        &format!("{:?}", error_receipt.display().to_string()),
    );
    let script = dir.path().join("parent.json");
    std::fs::write(
        &script,
        json!({
            "engine": "faux",
            "repeatLastResponse": true,
            "responses": [
                { "content": [
                    { "type": "toolCall", "name": "python_repl", "arguments": { "code": later } },
                ] },
                { "text": "noted" },
            ],
        })
        .to_string(),
    )
    .expect("write parent script");
    let child_script = dir.path().join("child.json");
    std::fs::write(
        &child_script,
        json!({ "engine": "faux", "responses": [{ "text": "done" }] }).to_string(),
    )
    .expect("write child script");

    let socket = dir.path().join("idle.sock");
    let _daemon = spawn_supervisor(&socket, &agent_dir, &kernel_python);
    wait_socket_ready(&socket);
    let (mut client, hello) = Client::connect(&socket);
    assert_eq!(hello["type"], "daemon_hello");
    client.send_command(
        "c1",
        &json!({
            "type": "create",
            "config": {
                "cwd": dir.path().to_string_lossy(),
                "sessionDir": sessions_dir.to_string_lossy(),
                "script": script.to_string_lossy(),
                "childScript": child_script.to_string_lossy(),
            },
        }),
    );
    let created = client.read_response("c1");
    assert_eq!(created["success"], true, "create failed: {created}");
    let session_id = created["data"]["activeSessionId"]
        .as_str()
        .or_else(|| created["data"]["id"].as_str())
        .expect("active session id")
        .to_string();
    client.send_command(
        "p1",
        &json!({ "type": "prompt", "activeSessionId": session_id, "message": "spawn later" }),
    );
    assert_eq!(client.read_response("p1")["success"], true, "prompt failed");

    // Time from the spawn's ledger row to the child's task row, both polled.
    let ledger_dir = agent_dir.join("rlm-ledger");
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut spawned: Option<(Instant, String)> = None;
    let waited = loop {
        if let Ok(error) = std::fs::read_to_string(&error_receipt) {
            panic!("the background spawn failed: {error}");
        }
        if spawned.is_none() {
            spawned = std::fs::read_dir(&ledger_dir)
                .into_iter()
                .flatten()
                .flatten()
                .flat_map(|entry| session_rows(&entry.path().to_string_lossy()))
                .find(|row| row["op"] == "spawn")
                .and_then(|row| Some((Instant::now(), row["child"].as_str()?.to_string())));
        }
        if let Some((at, child)) = &spawned {
            let has_task = std::fs::read_to_string(child).is_ok_and(|_| {
                session_rows(child)
                    .iter()
                    .any(|row| row["customType"] == "agent_message")
            });
            if has_task {
                break at.elapsed();
            }
        }
        assert!(
            Instant::now() < deadline,
            "the child never got its task (spawned: {spawned:?})"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(
        waited < Duration::from_secs(10),
        "an idle parent's spawn waited {waited:?} for its task"
    );
}
