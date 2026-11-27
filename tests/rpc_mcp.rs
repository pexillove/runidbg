#![cfg(all(feature = "rpc", feature = "mcp"))]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;

use runidbg_arm64_emu_tracer::script::mcp::McpServer;
use runidbg_arm64_emu_tracer::script::rpc::{start_rpc_server, RpcServerConfig, RpcSource};
use serde_json::{json, Value};

fn send_http(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, Value) {
    let mut stream = TcpStream::connect(addr).expect("connect rpc server");
    let body_bytes = body.unwrap_or("").as_bytes();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(req.as_bytes()).expect("write headers");
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes).expect("write body");
    }
    stream.flush().expect("flush");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8(raw).expect("utf8 response");
    let (head, body_str) = text.split_once("\r\n\r\n").expect("split http response");
    let status_line = head.lines().next().expect("status line");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("parse status code");
    let parsed: Value = serde_json::from_str(body_str)
        .unwrap_or_else(|e| panic!("invalid JSON body ({e}): {body_str}"));
    (status, parsed)
}

#[test]
fn test_rpc_server_http_and_jsonrpc() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = r#"
const emu = Emu.create();
const mod = emu.load("samples/libstatic-webp.so");

rpc.exports = {
    version() {
        return mod.call("WebPGetDecoderVersion");
    },
    echo(a, b) {
        return { a, b, sum: a + b };
    },
};
"#;

    let handle = start_rpc_server(
        RpcSource::Inline {
            code: script.to_string(),
            base_dir: root,
        },
        RpcServerConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            allow_eval: true,
            ..Default::default()
        },
    )
    .expect("start_rpc_server");

    let addr = handle.local_addr();
    assert_eq!(handle.workers(), 1);

    // 1. GET /health
    let (status, health) = send_http(addr, "GET", "/health", None);
    assert_eq!(status, 200);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["workers"], 1);
    assert!(health["exports"].is_array());

    // 2. GET /exports
    let (status, exports) = send_http(addr, "GET", "/exports", None);
    assert_eq!(status, 200);
    let list = exports["exports"].as_array().expect("exports array");
    assert!(list.iter().any(|v| v == "version"));
    assert!(list.iter().any(|v| v == "echo"));

    // 3. POST /call/version
    let (status, ver_resp) = send_http(addr, "POST", "/call/version", Some("[]"));
    assert_eq!(status, 200);
    let ver = ver_resp["result"].as_u64().expect("version u64");
    assert!(ver > 0, "WebPGetDecoderVersion should be > 0, got {ver}");

    // 4. POST /rpc (JSON-RPC 2.0 rpc.list)
    let rpc_list_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "rpc.list"
    })
    .to_string();
    let (status, rpc_list_resp) = send_http(addr, "POST", "/rpc", Some(&rpc_list_req));
    assert_eq!(status, 200);
    assert_eq!(rpc_list_resp["id"], 1);
    assert!(rpc_list_resp["result"]
        .as_array()
        .expect("rpc.list result")
        .iter()
        .any(|v| v == "echo"));

    // 5. POST /rpc (JSON-RPC 2.0 echo)
    let rpc_echo_req = json!({
        "jsonrpc": "2.0",
        "id": "req-2",
        "method": "echo",
        "params": [17, 25]
    })
    .to_string();
    let (status, rpc_echo_resp) = send_http(addr, "POST", "/rpc", Some(&rpc_echo_req));
    assert_eq!(status, 200);
    assert_eq!(rpc_echo_resp["id"], "req-2");
    assert_eq!(rpc_echo_resp["result"]["sum"], 42);

    // 6. POST /eval
    let eval_req = json!({
        "code": "console.log('rpc-eval-log'); 6 * 7"
    })
    .to_string();
    let (status, eval_resp) = send_http(addr, "POST", "/eval", Some(&eval_req));
    assert_eq!(status, 200);
    assert_eq!(eval_resp["result"], 42);
    assert_eq!(eval_resp["logs"], json!(["[log] rpc-eval-log"]));

    // 7. Unknown RPC method returns error
    let (status, err_resp) = send_http(addr, "POST", "/call/not_found", Some("[]"));
    assert_eq!(status, 500);
    assert!(err_resp["error"].is_string());

    handle.shutdown();
}

#[test]
fn test_rpc_multi_worker_batch_and_async_exports() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = r#"
const emu = Emu.create();
const mod = emu.load("samples/libstatic-webp.so");
const snap = emu.snapshot();

rpc.exports = {
    async asyncVersion(tag) {
        emu.restore(snap);
        const v = await Promise.resolve(mod.call("WebPGetDecoderVersion"));
        return { tag, version: v };
    },
    mul(a, b) {
        return a * b;
    },
};
"#;

    let handle = start_rpc_server(
        RpcSource::Inline {
            code: script.to_string(),
            base_dir: root,
        },
        RpcServerConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            workers: 4,
            max_queue: 256,
            timeout_ms: 10_000,
            ..Default::default()
        },
    )
    .expect("start multi-worker rpc server");

    let addr = handle.local_addr();
    assert_eq!(handle.workers(), 4);

    // 1. /health reports 4 workers
    let (status, health) = send_http(addr, "GET", "/health", None);
    assert_eq!(status, 200);
    assert_eq!(health["workers"], 4);

    // 2. async rpc.exports method works
    let (status, async_resp) = send_http(addr, "POST", "/call/asyncVersion", Some(r#"["t0"]"#));
    assert_eq!(status, 200);
    assert_eq!(async_resp["result"]["tag"], "t0");
    assert!(async_resp["result"]["version"].as_u64().unwrap() > 0);

    // 3. JSON-RPC 2.0 Batch fans out across worker pool and preserves response order
    let batch: Vec<Value> = (0..12)
        .map(|i| {
            json!({
                "jsonrpc": "2.0",
                "id": i,
                "method": "mul",
                "params": [i, 3]
            })
        })
        .collect();
    let batch_body = serde_json::to_string(&batch).unwrap();
    let (status, batch_resp) = send_http(addr, "POST", "/rpc", Some(&batch_body));
    assert_eq!(status, 200);
    let arr = batch_resp.as_array().expect("batch response array");
    assert_eq!(arr.len(), 12);
    for (i, item) in arr.iter().enumerate() {
        assert_eq!(item["id"], i);
        assert_eq!(item["result"], i * 3);
    }

    // 4. Concurrent multi-client stress test (8 threads x 2 requests = 16 parallel calls)
    let mut threads = Vec::new();
    for t in 0..8usize {
        threads.push(std::thread::spawn(move || {
            for k in 0..2usize {
                let tag = format!("t{t}-k{k}");
                let body = serde_json::to_string(&json!([tag])).unwrap();
                let (st, resp) = send_http(addr, "POST", "/call/asyncVersion", Some(&body));
                assert_eq!(st, 200);
                assert_eq!(resp["result"]["tag"], tag);
                assert!(resp["result"]["version"].as_u64().unwrap() > 0);
            }
        }));
    }
    for th in threads {
        th.join().expect("client thread panic");
    }

    handle.shutdown();
}

fn send_http_raw(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("connect rpc server");
    let body_bytes = body.unwrap_or("").as_bytes();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(req.as_bytes()).expect("write headers");
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes).expect("write body");
    }
    stream.flush().expect("flush");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8(raw).expect("utf8 response");
    let (head, body_str) = text.split_once("\r\n\r\n").expect("split http response");
    let status: u16 = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (status, body_str.to_string())
}

#[test]
fn test_rpc_phase1_hard_timeout_metrics_and_reload() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "runidbg_p1_test_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let script_path = tmp_dir.join("service.js");

    let root_str = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples/libstatic-webp.so")
        .to_string_lossy()
        .into_owned();
    // v1 script: exposes version() and infiniteLoop() (ARM64 `b #0` = 0x14000000 in executable .text)
    let v1_js = format!(
        r#"
const emu = Emu.create();
const mod = emu.load({root_str:?});
const fnAddr = mod.symbol("WebPGetDecoderVersion");
emu.mem.patch(fnAddr, new Uint8Array([0x00, 0x00, 0x00, 0x14]));
const snap = emu.snapshot();

rpc.exports = {{
    version() {{ return "v1"; }},
    infiniteLoop() {{
        emu.restore(snap);
        return emu.call(fnAddr);
    }},
}};
"#
    );
    std::fs::write(&script_path, v1_js).expect("write v1 script");

    let handle = start_rpc_server(
        RpcSource::File(script_path.clone()),
        RpcServerConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            workers: 2,
            timeout_ms: 5_000,
            max_instructions: 64,
            allow_reload: true,
            ..Default::default()
        },
    )
    .expect("start phase1 rpc server");

    let addr = handle.local_addr();

    // 1. Normal call returns v1
    let (st, resp) = send_http(addr, "POST", "/call/version", Some("[]"));
    assert_eq!(st, 200);
    assert_eq!(resp["result"], "v1");

    // 2. Infinite loop is hard-killed by Unicorn max_instructions -> HTTP 504
    let (st, timeout_resp) = send_http(addr, "POST", "/call/infiniteLoop", Some("[]"));
    assert_eq!(st, 504, "expected 504 on instruction budget exceeded: {timeout_resp}");

    // 3. Worker immediately recovers and serves subsequent requests
    let (st, resp2) = send_http(addr, "POST", "/call/version", Some("[]"));
    assert_eq!(st, 200);
    assert_eq!(resp2["result"], "v1");

    // 4. GET /metrics (JSON + Prometheus)
    let (st, metrics_json) = send_http(addr, "GET", "/metrics", None);
    assert_eq!(st, 200);
    assert_eq!(metrics_json["workers"], 2);
    assert!(metrics_json["requests_ok"].as_u64().unwrap() >= 2);
    assert!(metrics_json["requests_timeout"].as_u64().unwrap() >= 1);

    let (st, prom_text) = send_http_raw(addr, "GET", "/metrics?format=prom", None);
    assert_eq!(st, 200);
    assert!(prom_text.contains("runidbg_rpc_workers 2"));
    assert!(prom_text.contains("runidbg_rpc_requests_total{status=\"timeout\"} 1"));

    // 5. Hot reload to v2
    let v2_js = r#"
rpc.exports = {
    version() { return "v2"; },
};
"#;
    std::fs::write(&script_path, v2_js).expect("write v2 script");
    let (st, reload_resp) = send_http(addr, "POST", "/admin/reload", None);
    assert_eq!(st, 200);
    assert_eq!(reload_resp["ok"], true);
    assert_eq!(reload_resp["reload_epoch"], 1);

    let (st, v2_resp) = send_http(addr, "POST", "/call/version", Some("[]"));
    assert_eq!(st, 200);
    assert_eq!(v2_resp["result"], "v2");

    // 6. Broken v3 script is rejected on reload without breaking live v2 workers
    std::fs::write(&script_path, "this is broken syntax {{{").expect("write broken v3");
    let (st, bad_reload) = send_http(addr, "POST", "/admin/reload", None);
    assert_eq!(st, 500);
    assert_eq!(bad_reload["ok"], false);

    let (st, still_v2) = send_http(addr, "POST", "/call/version", Some("[]"));
    assert_eq!(st, 200);
    assert_eq!(still_v2["result"], "v2");

    handle.shutdown();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

fn mcp_call(server: &mut McpServer, id: u64, name: &str, arguments: Value) -> Value {
    let req = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    })
    .to_string();
    let raw = server
        .handle_message(&req)
        .expect("tools/call must return a response");
    let resp: Value = serde_json::from_str(&raw).expect("valid json response");
    assert_eq!(resp["id"], id);
    assert!(
        resp.get("error").is_none(),
        "unexpected JSON-RPC error: {resp}"
    );
    let result = &resp["result"];
    assert_eq!(
        result["isError"], false,
        "MCP tool `{name}` returned error: {result}"
    );
    let text = result["content"][0]["text"]
        .as_str()
        .expect("content[0].text");
    serde_json::from_str(text).expect("tool payload json")
}

#[test]
fn test_mcp_server_protocol_and_tools() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut server = McpServer::new(&root, None).expect("McpServer::new");

    // 1. initialize
    let init_raw = server
        .handle_message(
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": "2024-11-05" }
            })
            .to_string(),
        )
        .expect("initialize response");
    let init: Value = serde_json::from_str(&init_raw).unwrap();
    assert_eq!(init["result"]["serverInfo"]["name"], "runidbg");
    assert!(init["result"]["capabilities"]["tools"].is_object());

    // 2. notifications/initialized -> None
    let notif = server.handle_message(
        &json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
        .to_string(),
    );
    assert!(notif.is_none());

    // 3. ping
    let ping_raw = server
        .handle_message(&json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" }).to_string())
        .expect("ping response");
    let ping: Value = serde_json::from_str(&ping_raw).unwrap();
    assert_eq!(ping["result"], json!({}));

    // 4. tools/list
    let list_raw = server
        .handle_message(&json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/list" }).to_string())
        .expect("tools/list response");
    let list: Value = serde_json::from_str(&list_raw).unwrap();
    let tools = list["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 39);
    assert!(tools.iter().any(|t| t["name"] == "trace_calltree"));
    assert!(tools.iter().any(|t| t["name"] == "trace_crypto_scan"));
    assert!(tools.iter().any(|t| t["name"] == "trace_taint"));
    assert!(tools.iter().any(|t| t["name"] == "trace_decompile"));
    assert!(tools.iter().any(|t| t["name"] == "trace_mem"));
    assert!(tools.iter().any(|t| t["name"] == "emu_asm"));
    assert!(tools.iter().any(|t| t["name"] == "trace_query"));
    assert!(tools.iter().any(|t| t["name"] == "trace_diff"));
    assert!(tools.iter().any(|t| t["name"] == "trace_vm_analyze"));
    assert!(tools.iter().any(|t| t["name"] == "emu_save_tape"));
    assert!(tools.iter().any(|t| t["name"] == "emu_load_tape"));
    assert!(tools.iter().any(|t| t["name"] == "emu_seek_tape"));
    assert!(tools.iter().any(|t| t["name"] == "tape_info"));
    assert!(tools.iter().any(|t| t["name"] == "tt_record_start"));
    assert!(tools.iter().any(|t| t["name"] == "tt_record_stop"));
    assert!(tools.iter().any(|t| t["name"] == "tt_open"));
    assert!(tools.iter().any(|t| t["name"] == "tt_seek"));
    assert!(tools.iter().any(|t| t["name"] == "tt_reverse_watch"));

    // 5. emu_create -> emu_load -> emu_exports -> emu_call -> emu_mem_read -> emu_mem_scan -> emu_regs -> emu_eval_js
    let created = mcp_call(&mut server, 10, "emu_create", json!({}));
    let emu_id = created["emuId"].as_u64().expect("emuId");

    let loaded = mcp_call(
        &mut server,
        11,
        "emu_load",
        json!({ "emuId": emu_id, "path": "samples/libstatic-webp.so" }),
    );
    let mod_id = loaded["modId"].as_u64().expect("modId");
    let base = loaded["base"].as_u64().expect("base");
    assert!(base > 0);

    let exports = mcp_call(
        &mut server,
        12,
        "emu_exports",
        json!({ "modId": mod_id, "filter": "WebPGetDecoderVersion" }),
    );
    assert!(exports["total"].as_u64().unwrap() >= 1);
    let fn_addr_eval = mcp_call(
        &mut server,
        120,
        "emu_eval_js",
        json!({ "code": format!("__mcp.mods[{mod_id}].mod.symbol('WebPGetDecoderVersion')") }),
    );
    let fn_addr = fn_addr_eval["result"].as_u64().expect("fn addr");

    let call_res = mcp_call(
        &mut server,
        13,
        "emu_call",
        json!({ "emuId": emu_id, "modId": mod_id, "target": "WebPGetDecoderVersion" }),
    );
    let retval = call_res["retval"].as_u64().expect("retval");
    assert!(retval > 0);

    let mem = mcp_call(
        &mut server,
        14,
        "emu_mem_read",
        json!({ "emuId": emu_id, "addr": format!("0x{base:x}"), "len": 4, "format": "hex" }),
    );
    assert_eq!(mem["hex"], "7f454c46");

    let scan = mcp_call(
        &mut server,
        15,
        "emu_mem_scan",
        json!({ "emuId": emu_id, "start": base, "len": 64, "pattern": "7f 45 4c 46" }),
    );
    assert!(scan["total"].as_u64().unwrap() >= 1);

    let regs = mcp_call(&mut server, 16, "emu_regs", json!({ "emuId": emu_id }));
    assert!(regs["pc"].is_string() || regs["pc"].is_number());
    assert!(regs["sp"].is_string() || regs["sp"].is_number());

    let eval = mcp_call(
        &mut server,
        17,
        "emu_eval_js",
        json!({ "code": "console.log('hello-mcp'); 40 + 2" }),
    );
    assert_eq!(eval["result"], 42);
    assert_eq!(eval["logs"], json!(["[log] hello-mcp"]));

    // 6. Phase 3: emu_disasm
    let disasm = mcp_call(
        &mut server,
        18,
        "emu_disasm",
        json!({ "emuId": emu_id, "addr": format!("0x{fn_addr:x}"), "count": 4 }),
    );
    assert_eq!(disasm["count"], 4);
    assert!(disasm["instructions"][0]["asm"].is_string());

    // 7. Phase 3: emu_trace_call
    let traced = mcp_call(
        &mut server,
        19,
        "emu_trace_call",
        json!({ "emuId": emu_id, "modId": mod_id, "target": "WebPGetDecoderVersion" }),
    );
    assert_eq!(traced["retval"], retval);
    assert!(traced["trace"]["records"].as_u64().unwrap() >= 1);

    // 8. Phase 3: emu_hook_capture
    let hooked = mcp_call(
        &mut server,
        20,
        "emu_hook_capture",
        json!({ "emuId": emu_id, "modId": mod_id, "hookAddr": fn_addr, "callTarget": "WebPGetDecoderVersion" }),
    );
    assert_eq!(hooked["hits"], 1);
    assert_eq!(hooked["retval"], retval);

    // 9. Phase 3: emu_diag_missing
    let diag_en = mcp_call(
        &mut server,
        21,
        "emu_diag_missing",
        json!({ "emuId": emu_id, "action": "enable" }),
    );
    assert_eq!(diag_en["enabled"], true);
    let diag_rep = mcp_call(
        &mut server,
        22,
        "emu_diag_missing",
        json!({ "emuId": emu_id, "action": "report" }),
    );
    assert!(diag_rep["patchJs"].is_string());

    // 9.5. Phase 3: trace_calltree & trace_crypto_scan via offline trace file
    let tmp_idx = std::env::temp_dir().join("runidbg_mcp_trace_test.idx");
    let tmp_idx_str = tmp_idx.to_string_lossy().to_string();
    let _ = mcp_call(
        &mut server,
        23,
        "emu_eval_js",
        json!({
            "code": format!(
                "(() => {{ \
                    const e = __mcp.emus[{emu_id}]; \
                    const m = __mcp.mods[{mod_id}].mod; \
                    e.trace.start({{ output: {:?}, level: 'full', start: m.base, end: m.base + m.size, capacity: 4096 }}); \
                    m.call('WebPGetDecoderVersion'); \
                    return e.trace.stop(); \
                }})()",
                tmp_idx_str
            )
        }),
    );

    let calltree = mcp_call(
        &mut server,
        24,
        "trace_calltree",
        json!({ "idxPath": tmp_idx_str, "base": format!("0x{base:x}"), "maxDepth": 4, "minInsns": 1 }),
    );
    assert!(calltree["totalRecords"].as_u64().unwrap() >= 1);
    assert!(calltree["returned"].as_u64().unwrap() >= 1);
    assert!(calltree["nodes"][0]["targetOffset"].is_string());

    let crypto = mcp_call(
        &mut server,
        25,
        "trace_crypto_scan",
        json!({ "idxPath": tmp_idx_str, "base": format!("0x{base:x}") }),
    );
    assert!(crypto["totalRecords"].as_u64().unwrap() >= 1);
    assert!(crypto["groups"].is_array());

    let inspected = mcp_call(
        &mut server,
        26,
        "trace_inspect",
        json!({ "idxPath": tmp_idx_str, "base": format!("0x{base:x}"), "action": "dump", "limit": 5 }),
    );
    assert!(inspected["returned"].as_u64().unwrap() >= 1);

    let _ = std::fs::remove_file(&tmp_idx);
    let _ = std::fs::remove_file(tmp_idx.with_extension("ext"));

    // 9.5 MCP Tape 工具链验证 (emu_save_tape -> tape_info -> emu_load_tape -> emu_exports/emu_call -> emu_seek_tape)
    let tmp_tape = std::env::temp_dir().join(format!("runidbg_mcp_tape_{}.rtape", std::process::id()));
    let tmp_tape_str = tmp_tape.to_string_lossy().to_string();
    // 缺省 label 应回退为 "base"，而不是 "null"
    let saved_tape = mcp_call(
        &mut server,
        27,
        "emu_save_tape",
        json!({ "emuId": emu_id, "path": tmp_tape_str }),
    );
    assert_eq!(saved_tape["checkpoints"].as_array().unwrap().len(), 1);
    assert_eq!(saved_tape["checkpoints"][0]["label"], "base");
    let cp1 = mcp_call(
        &mut server,
        28,
        "emu_save_tape",
        json!({ "emuId": emu_id, "path": tmp_tape_str, "label": "mcp_cp1", "append": true }),
    );
    assert_eq!(cp1["label"], "mcp_cp1");
    assert_eq!(cp1["isDelta"], true);
    let tape_info = mcp_call(&mut server, 29, "tape_info", json!({ "path": tmp_tape_str }));
    assert_eq!(tape_info["checkpoints"].as_array().unwrap().len(), 2);
    let loaded_tape = mcp_call(
        &mut server,
        30,
        "emu_load_tape",
        json!({ "path": tmp_tape_str, "checkpoint": 0, "backend": null }),
    );
    let tape_emu_id = loaded_tape["emuId"].as_u64().expect("tape emuId");
    let loaded_mods = loaded_tape["modules"].as_array().expect("loaded modules");
    assert_eq!(loaded_mods.len(), 1);
    assert!(loaded_mods[0]["base"].as_u64().unwrap() > 0);
    assert!(loaded_mods[0]["exportCount"].as_u64().unwrap() >= 1);
    let restored_mod_id = loaded_mods[0]["modId"].as_u64().expect("restored modId");
    let restored_exports = mcp_call(
        &mut server,
        301,
        "emu_exports",
        json!({ "modId": restored_mod_id, "filter": "WebPGetDecoderVersion" }),
    );
    assert!(restored_exports["total"].as_u64().unwrap() >= 1);
    let sought = mcp_call(
        &mut server,
        31,
        "emu_seek_tape",
        json!({ "emuId": tape_emu_id, "path": tmp_tape_str, "checkpoint": "mcp_cp1" }),
    );
    assert_eq!(sought["label"], "mcp_cp1");
    assert_eq!(sought["loadedModules"].as_array().unwrap().len(), 1);
    let _ = std::fs::remove_file(&tmp_tape);

    // 9.6 MCP Phase 2 Time-Travel 工具链验证 (tt_record_start -> emu_call -> tt_record_stop -> tt_open -> tt_seek -> tt_reverse_watch -> tt_mem_read -> tt_close)
    let tmp_tt = std::env::temp_dir().join(format!("runidbg_mcp_tt_{}.rtape", std::process::id()));
    let tmp_tt_str = tmp_tt.to_string_lossy().to_string();
    let rec_start = mcp_call(
        &mut server,
        32,
        "tt_record_start",
        json!({ "emuId": emu_id, "keyframeInterval": null, "captureJni": null }),
    );
    assert_eq!(rec_start["ok"], true);
    let _ = mcp_call(
        &mut server,
        33,
        "emu_call",
        json!({ "emuId": emu_id, "target": fn_addr, "args": [] }),
    );
    let rec_stop = mcp_call(
        &mut server,
        34,
        "tt_record_stop",
        json!({ "emuId": emu_id, "path": tmp_tt_str, "bookmark": "webp_end" }),
    );
    assert!(rec_stop["timeline"]["endInsn"].as_u64().unwrap() >= 1);
    assert_eq!(rec_stop["timeline"]["keyframeInterval"].as_u64(), Some(100_000));
    let tt_opened = mcp_call(&mut server, 35, "tt_open", json!({ "path": tmp_tt_str }));
    let tt_id = tt_opened["ttId"].as_u64().expect("ttId");
    let total_insns = tt_opened["totalInsns"].as_u64().unwrap();
    assert!(total_insns >= 1);
    let tt_at_end = mcp_call(
        &mut server,
        36,
        "tt_seek",
        json!({ "ttId": tt_id, "bookmark": "webp_end" }),
    );
    assert_eq!(tt_at_end["stop"]["insn"].as_u64(), Some(total_insns));
    let tt_noop = mcp_call(
        &mut server,
        361,
        "tt_seek",
        json!({ "ttId": tt_id, "delta": 0 }),
    );
    assert_eq!(tt_noop["stop"]["insn"].as_u64(), Some(total_insns));
    let tt_back = mcp_call(
        &mut server,
        37,
        "tt_seek",
        json!({ "ttId": tt_id, "delta": -1 }),
    );
    assert_eq!(tt_back["stop"]["insn"].as_u64(), Some(total_insns - 1));
    // tt_reverse_watch 支持 0x 十六进制地址字符串且未命中时保持当前 insn 不变
    let fn_hex = format!("0x{fn_addr:x}");
    let rw_miss = mcp_call(
        &mut server,
        38,
        "tt_reverse_watch",
        json!({ "ttId": tt_id, "addr": fn_hex, "len": 4, "kind": "write" }),
    );
    assert!(rw_miss["hit"].is_null());
    assert_eq!(rw_miss["currentInsn"].as_u64(), Some(total_insns - 1));
    // tt_mem_read 读取当前指令点内存
    let mem_res = mcp_call(
        &mut server,
        39,
        "tt_mem_read",
        json!({ "ttId": tt_id, "addr": fn_hex, "len": 16 }),
    );
    assert_eq!(mem_res["len"].as_u64(), Some(16));
    assert_eq!(mem_res["hex"].as_str().unwrap().len(), 32);
    // tt_close 释放时间旅行会话
    let closed = mcp_call(&mut server, 40, "tt_close", json!({ "ttId": tt_id }));
    assert_eq!(closed["closed"], true);
    let _ = std::fs::remove_file(&tmp_tt);

    // 10. Phase 3: Streamable HTTP MCP
    let http_mcp = runidbg_arm64_emu_tracer::script::mcp::start_mcp_http_server("127.0.0.1:0", &root, None)
        .expect("start_mcp_http_server");
    let mcp_addr = http_mcp.local_addr();
    let ping_req = json!({ "jsonrpc": "2.0", "id": 99, "method": "ping" }).to_string();
    let (st, http_ping) = send_http(mcp_addr, "POST", "/mcp", Some(&ping_req));
    assert_eq!(st, 200);
    assert_eq!(http_ping["id"], 99);
    assert_eq!(http_ping["result"], json!({}));
    http_mcp.shutdown();
}
