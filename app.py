import json
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

from flask import Flask, Response, jsonify, render_template_string
from flask import request as flask_request, send_from_directory, stream_with_context

import ai_engine
import db as database
import facts
import few_shot
import file_gen
import math_tool
import memory as mem
import query_expand
import reasoning
import web_search
from db import UPLOAD_DIR, AGENT_FILES_DIR
from personality import get_personality, list_personalities
from utils import anonymize_for_cloud, summarize_title, similarity, tokenize, now_iso

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None

CONTEXT_LIMIT_LOCAL = ai_engine.OLLAMA_NUM_CTX
CONTEXT_LIMIT_CLOUD = 32768

MEMORY_CONTEXT_TOKEN_BUDGET = 1300
HISTORY_CONTEXT_TOKEN_BUDGET = 1800
CLOUD_MEMORY_CONTEXT_TOKEN_BUDGET = 4200
CLOUD_HISTORY_CONTEXT_TOKEN_BUDGET = 5200
LOCAL_RETRIEVAL_LIMIT = 10
LOCAL_RETRIEVAL_CHAR_BUDGET = 5200
CLOUD_RETRIEVAL_LIMIT = 18
CLOUD_RETRIEVAL_CHAR_BUDGET = 15000
LOCAL_ATTACHMENT_CHARS = 5200
CLOUD_ATTACHMENT_CHARS = 18000
LOCAL_KNOWLEDGE_LIMIT = 3
LOCAL_KNOWLEDGE_CHAR_BUDGET = 2400
CLOUD_KNOWLEDGE_LIMIT = 6
CLOUD_KNOWLEDGE_CHAR_BUDGET = 7000

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Marcellus</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f1f3f8;
      --panel: #ffffff;
      --panel-2: #eef1f7;
      --line: #dfe5ef;
      --text: #1c2434;
      --muted: #5f6c83;
      --accent: #4055d4;
      --accent-h: #3242ad;
      --accent-fg: #ffffff;
      --accent-soft: rgba(64,85,212,.12);
      --danger: #dc2626;
      --green: #10b981;
      --shadow: 0 6px 22px rgba(15,23,42,.06);
      --radius: 12px;
      --user-bg: #e3e9fb;   --user-line: #ccd6f6;
      --hover-bg: #e3e9fb;  --hover-line: #a9bcf5;
      --focus: #a9bcf5;
      --code-bg: #171e2e;   --code-fg: #e2e8f0;  --code-line: #2d3748;
      --icode-bg: #edf1f7;  --icode-fg: #35509c;
      --ok-bg: #dcfce7;     --ok-line: #86efac;
      --bad-bg: #fee2e2;    --bad-line: #fca5a5;
      --warn-bg: #fef3c7;   --warn-fg: #92400e;  --warn-line: #fbbf24;
      --alert-bg: #fee2e2;  --alert-fg: #991b1b; --alert-line: #f87171;
      --zip-bg: #f5f3ff;    --zip-line: #a78bfa; --zip-hover: #ede9fe;
    }
    [data-theme="dark"] {
      color-scheme: dark;
      --bg: #0e1117;
      --panel: #161b24;
      --panel-2: #1d2430;
      --line: #2a3342;
      --text: #e6eaf2;
      --muted: #8b96ab;
      --accent: #e3a63c;
      --accent-h: #f0b954;
      --accent-fg: #201503;
      --accent-soft: rgba(227,166,60,.15);
      --shadow: 0 6px 22px rgba(0,0,0,.35);
      --user-bg: #263041;   --user-line: #35415a;
      --hover-bg: #263041;  --hover-line: #4a5875;
      --focus: #b98a2e;
      --code-bg: #10151f;   --code-fg: #dbe2ee;  --code-line: #262f3f;
      --icode-bg: #232b3a;  --icode-fg: #ecc069;
      --ok-bg: #16341f;     --ok-line: #2c6e42;
      --bad-bg: #3a1a1a;    --bad-line: #7f2f2f;
      --warn-bg: #33270d;   --warn-fg: #f3c96b;  --warn-line: #8a6414;
      --alert-bg: #391a1a;  --alert-fg: #f3a1a1; --alert-line: #a03535;
      --zip-bg: #241f36;    --zip-line: #6b5bb8; --zip-hover: #2c2545;
    }
    *::-webkit-scrollbar { width: 9px; height: 9px; }
    *::-webkit-scrollbar-thumb { background: var(--line); border-radius: 8px; }
    *::-webkit-scrollbar-track { background: transparent; }
    body, .topbar, .composer, .message, .ctrl-row select, .ctrl-row input, textarea {
      transition: background-color .22s ease, color .22s ease, border-color .22s ease;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }
    button, input, textarea, select { font: inherit; }

    /* Layout */
    .app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }

    /* Sidebar */
    .sidebar {
      background: linear-gradient(160deg, #0f1729 0%, #111827 100%);
      color: #f1f5f9;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 16px;
      overflow: hidden;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 4px 0 8px;
      border-bottom: 1px solid rgba(255,255,255,.08);
    }
    .brand-name { font-size: 19px; font-weight: 700; letter-spacing: -.3px; }
    .brand-tag { font-size: 11px; color: #94a3b8; margin-top: 1px; }
    .btn-new {
      border: 0;
      background: var(--accent);
      color: var(--accent-fg);
      border-radius: var(--radius);
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      transition: background .15s;
    }
    .btn-new:hover { background: var(--accent-h); }
    .history { overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 6px; }
    .conv {
      border: 1px solid rgba(255,255,255,.07);
      background: rgba(255,255,255,.05);
      border-radius: 8px;
      padding: 8px 10px;
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 5px;
      align-items: center;
    }
    .conv.active { background: rgba(79,110,247,.25); border-color: rgba(147,197,253,.35); }
    .conv-title {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      cursor: pointer; font-size: 13.5px; min-width: 0;
    }
    .conv-personality { font-size: 14px; }
    .icon-btn {
      border: 0; border-radius: 6px;
      background: rgba(255,255,255,.08);
      color: inherit; width: 28px; height: 28px;
      cursor: pointer; font-size: 13px;
    }
    .icon-btn:hover { background: rgba(255,255,255,.16); }
    .icon-btn.danger:hover { background: rgba(220,38,38,.5); }
    .status-bar {
      font-size: 11.5px; color: #94a3b8;
      border-top: 1px solid rgba(255,255,255,.1);
      padding-top: 10px;
      line-height: 1.5;
    }
    .mem-badge { color: #a5b4fc; font-weight: 500; }

    /* Main */
    .main { display: grid; grid-template-rows: auto 1fr auto; height: 100vh; min-width: 0; }

    /* Topbar */
    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 12px 20px;
      display: flex; align-items: center; gap: 10px;
      flex-wrap: wrap;
    }
    .topbar-title { font-size: 15px; font-weight: 600; flex: 1; min-width: 120px; }
    .ctrl-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .ctrl-label { font-size: 12px; color: var(--muted); white-space: nowrap; }
    .ctrl-row select, .ctrl-row input {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      background: var(--panel-2);
      color: var(--text);
      font-size: 13px;
    }
    .ctrl-row select { min-width: 150px; }
    #apiKey { min-width: 200px; }
    .global-search {
      margin: 6px 10px; padding: 7px 10px; font-size: 12.5px;
      border: 1px solid var(--line); border-radius: 8px;
      background: var(--panel-2); color: var(--text); width: calc(100% - 20px);
    }
    .search-hit { cursor: pointer; padding: 8px 10px; border-radius: 8px; }
    .search-hit:hover { background: var(--panel-2); }
    .search-hit .sh-title { font-size: 12.5px; font-weight: 700; }
    .search-hit .sh-snippet { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
    .privacy-row { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }
    .privacy-row input { min-width: 0; width: 14px; height: 14px; }
    .workspace-row {
      display: flex; align-items: center; gap: 7px;
      flex: 1 1 420px; min-width: 260px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .workspace-row input {
      min-width: 180px; flex: 1;
      border: 1px solid var(--line); border-radius: 8px;
      padding: 6px 10px; background: var(--panel-2);
      color: var(--text); font-size: 13px;
    }
    .workspace-btn {
      border: 1px solid var(--line); background: var(--panel-2);
      color: var(--text); border-radius: 8px;
      min-width: 34px; height: 32px; padding: 0 9px;
      cursor: pointer; font-size: 13px;
    }
    .workspace-btn:hover { border-color: var(--hover-line); background: var(--hover-bg); }
    .workspace-status {
      color: var(--muted); font-size: 12px;
      white-space: nowrap; max-width: 260px;
      overflow: hidden; text-overflow: ellipsis;
    }
    .model-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #94a3b8; display: inline-block; margin-right: 4px;
    }
    .model-dot.loaded { background: var(--green); }

    /* Messages */
    .messages { overflow-y: auto; padding: 26px 22px; display: flex; flex-direction: column; gap: 16px; }
    .message {
      max-width: min(800px, 90%);
      padding: 14px 17px;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      line-height: 1.52;
      overflow-wrap: anywhere;
    }
    .message.user { align-self: flex-end; background: var(--user-bg); border: 1px solid var(--user-line); box-shadow: none; }
    .message.assistant { align-self: flex-start; background: var(--panel); border: 1px solid var(--line); }
    .meta { display: block; font-size: 11px; color: var(--muted); margin-bottom: 5px; font-weight: 700; text-transform: uppercase; }
    .attachments { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 5px; }
    .feedback { margin-top: 8px; display: flex; gap: 4px; opacity: .4; transition: opacity .15s; }
    .message.assistant:hover .feedback, .feedback:has(.active) { opacity: 1; }
    .fb-btn {
      border: 1px solid var(--line); background: var(--panel-2);
      border-radius: 8px; padding: 2px 8px; font-size: 13px;
      cursor: pointer; filter: grayscale(1);
    }
    .fb-btn:hover { filter: none; }
    .fb-btn.active { filter: none; }
    .fb-btn.active[data-v="up"] { background: var(--ok-bg); border-color: var(--ok-line); }
    .fb-btn.active[data-v="down"] { background: var(--bad-bg); border-color: var(--bad-line); }
    .pill {
      border: 1px solid var(--line); background: var(--panel-2);
      border-radius: 999px; padding: 4px 10px; font-size: 12px;
      text-decoration: none; color: var(--text);
      max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .code-block {
      background: var(--code-bg); color: var(--code-fg);
      border-radius: 8px; padding: 12px 14px;
      overflow-x: auto; margin: 8px 0;
      font-family: 'Cascadia Code', 'Fira Code', ui-monospace, monospace;
      font-size: 13px; line-height: 1.55;
      border: 1px solid var(--code-line);
      white-space: pre;
    }
    .code-lang { display: block; color: #64748b; font-size: 10px; margin-bottom: 6px; }
    .inline-code {
      background: var(--icode-bg); color: var(--icode-fg);
      padding: 1px 5px; border-radius: 4px;
      font-family: ui-monospace, monospace; font-size: .875em;
    }
    .empty { color: var(--muted); text-align: center; margin: auto; max-width: 380px; line-height: 1.6; }
    .thinking { color: var(--muted); font-style: italic; }

    /* Composer */
    .composer { background: var(--panel); border-top: 1px solid var(--line); padding: 14px 20px 18px; }
    .composer-inner { display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: end; }
    .file-label {
      width: 40px; height: 40px; display: grid; place-items: center;
      border: 1px solid var(--line); border-radius: 8px;
      background: var(--panel-2); cursor: pointer; font-size: 20px;
    }
    input[type="file"] { display: none; }
    textarea {
      resize: none; min-height: 40px; max-height: 150px;
      border: 1px solid var(--line); border-radius: 8px;
      padding: 10px 12px; outline: none; line-height: 1.45;
    }
    textarea:focus { border-color: var(--focus); box-shadow: 0 0 0 3px var(--accent-soft); }
    .btn-send {
      border: 0; background: var(--accent); color: var(--accent-fg);
      border-radius: 8px; padding: 10px 16px; cursor: pointer; font-weight: 600;
    }
    .btn-send:hover { background: var(--accent-h); }
    .btn-send.stop { background: var(--danger); }
    .file-list { margin: 8px 50px 0; color: var(--muted); font-size: 12px; display: flex; flex-wrap: wrap; gap: 5px; }

    /* Agent file cards */
    .agent-files-row {
      display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
    }
    .agent-file-card {
      display: flex; align-items: center; gap: 8px;
      border: 1px solid var(--line); background: var(--panel-2);
      border-radius: 8px; padding: 8px 14px;
      text-decoration: none; color: var(--text); font-size: 13px;
      transition: background .15s, border-color .15s;
      max-width: 260px;
    }
    .agent-file-card:hover { background: var(--hover-bg); border-color: var(--hover-line); }
    .agent-file-card .file-icon { font-size: 18px; flex-shrink: 0; }
    .agent-file-card .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    .agent-file-card .file-dl { color: var(--accent); font-size: 15px; flex-shrink: 0; }
    .agent-file-card.zip { border-color: var(--zip-line); background: var(--zip-bg); }
    .agent-file-card.zip:hover { background: var(--zip-hover); }

    /* Drop overlay */
    .drop-overlay {
      position: fixed; inset: 0; background: var(--accent-soft);
      border: 3px dashed var(--accent); z-index: 998;
      display: none; place-items: center; font-size: 26px;
      color: var(--accent); pointer-events: none;
    }
    .drop-overlay.active { display: grid; }

    /* Folder sidebar */
    .btn-folder {
      border: 1px dashed rgba(255,255,255,.2); background: transparent;
      color: #94a3b8; border-radius: 8px; padding: 7px 12px;
      cursor: pointer; font-size: 12px; width: 100%; transition: all .15s;
    }
    .btn-folder:hover { border-color: rgba(255,255,255,.4); color: #e2e8f0; }
    .folder-section { display: flex; flex-direction: column; gap: 3px; margin-top: 2px; }
    .folder-header {
      display: flex; align-items: center; gap: 5px;
      padding: 5px 6px; border-radius: 6px; cursor: pointer;
      font-size: 12px; font-weight: 600; color: #94a3b8; user-select: none;
    }
    .folder-header:hover { background: rgba(255,255,255,.06); }
    .folder-arrow { font-size: 9px; transition: transform .15s; flex-shrink: 0; }
    .folder-arrow.open { transform: rotate(90deg); }
    .folder-name-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .folder-body { display: flex; flex-direction: column; gap: 3px; padding-left: 8px; }
    .folder-body.collapsed { display: none; }

    /* Global conv context menu */
    .conv-menu-global {
      position: fixed; z-index: 999;
      background: #1a2540; border: 1px solid rgba(255,255,255,.14);
      border-radius: 10px; padding: 8px; box-shadow: 0 8px 28px rgba(0,0,0,.55);
      min-width: 210px; display: none; flex-direction: column; gap: 6px;
    }
    .conv-menu-global.open { display: flex; }
    .conv-menu-label { font-size: 11px; color: #64748b; padding: 2px 4px; }
    .conv-menu-global select {
      width: 100%; border-radius: 6px; border: 1px solid rgba(255,255,255,.15);
      background: #0f1729; color: #e2e8f0; padding: 5px 8px; font-size: 12.5px;
    }
    .conv-menu-del {
      border: 0; border-radius: 6px; background: transparent;
      color: #fca5a5; padding: 6px 8px; cursor: pointer; font-size: 12.5px; text-align: left;
    }
    .conv-menu-del:hover { background: rgba(220,38,38,.25); }

    /* Topbar title area */
    .topbar-title-area { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 120px; }
    .topbar-title { font-size: 15px; font-weight: 600; }
    #titleInput {
      flex: 1; font-size: 15px; font-weight: 600; border: 1px solid var(--line);
      border-radius: 6px; padding: 3px 8px; outline: none;
      background: var(--panel-2); color: var(--text); min-width: 120px;
    }
    #titleInput:focus { border-color: var(--focus); }

    @media (max-width: 720px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { display: none; }
      .message { max-width: 96%; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .composer-inner { grid-template-columns: auto 1fr; }
      .btn-send { grid-column: 1/-1; }
    }

    /* Agent file run button */
    .agent-file-wrap { display: inline-flex; align-items: center; gap: 6px; }
    .exec-agent-btn {
      border: 1px solid #2d4a6e; background: #1a3050;
      color: #60a5fa; border-radius: 6px;
      width: 30px; height: 30px; cursor: pointer; font-size: 13px;
      flex-shrink: 0; transition: background .15s;
    }
    .exec-agent-btn:hover { background: #1e4a7f; }
    .exec-agent-btn:disabled { opacity: .6; cursor: wait; }

    /* Code execution */
    .code-wrap { display: flex; flex-direction: column; }
    .exec-btn {
      align-self: flex-start; margin-top: 6px; padding: 5px 14px;
      background: #1a3050; color: #60a5fa;
      border: 1px solid #2d4a6e; border-radius: 6px;
      cursor: pointer; font-size: 12px; font-family: inherit;
      transition: background .15s;
    }
    .exec-btn:hover { background: #1e4a7f; }
    .exec-btn:disabled { opacity: .6; cursor: wait; }
    .exec-output {
      margin-top: 8px; padding: 10px 14px;
      background: #0d1424; border-radius: 6px;
      border-left: 3px solid #3b82f6;
      font-size: 12.5px; font-family: 'Cascadia Code', ui-monospace, monospace;
      white-space: pre-wrap; color: #e2e8f0;
      max-height: 1400px; overflow-y: auto; line-height: 1.5;
    }
    .exec-output.error { border-left-color: #ef4444; }
    .exec-send-btn {
      display: inline-block; margin-top: 8px; padding: 4px 12px;
      background: transparent; color: #60a5fa;
      border: 1px solid #2d4a6e; border-radius: 6px;
      cursor: pointer; font-size: 11.5px; font-family: inherit;
    }
    .exec-send-btn:hover { background: #1a3050; }
    .ctx-banner {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 16px; font-size: 13px; border-radius: 8px;
      margin: 0 0 8px 0; animation: fadeIn .3s ease;
    }
    .ctx-banner.warn  { background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-line); }
    .ctx-banner.alert { background: var(--alert-bg); color: var(--alert-fg); border: 1px solid var(--alert-line); }
    .ctx-banner a { color: inherit; font-weight: 700; cursor: pointer; text-decoration: underline; }
    .ctx-bar { flex: 1; height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
    .ctx-bar-fill { height: 100%; border-radius: 2px; transition: width .4s ease; }
    .ctx-banner.warn  .ctx-bar-fill { background: #f59e0b; }
    .ctx-banner.alert .ctx-bar-fill { background: #ef4444; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  </style>
</head>
<body>
<div id="dropOverlay" class="drop-overlay">📂 Solte os arquivos aqui</div>
<div id="convMenu" class="conv-menu-global">
  <span class="conv-menu-label">Mover para pasta</span>
  <select id="convFolderSel"></select>
  <button id="convDeleteBtn" class="conv-menu-del">🗑 Excluir conversa</button>
</div>
<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="brand">
      <svg width="26" height="22" viewBox="0 0 26 22" fill="none">
        <circle cx="3" cy="11" r="2.8" fill="#60a5fa"/>
        <circle cx="13" cy="3.5" r="2.8" fill="#a78bfa"/>
        <circle cx="13" cy="18.5" r="2.8" fill="#a78bfa"/>
        <circle cx="23" cy="11" r="2.8" fill="#60a5fa"/>
        <line x1="5.6" y1="9.8" x2="10.4" y2="5.2" stroke="#60a5fa" stroke-width="1.5" opacity=".65"/>
        <line x1="5.6" y1="12.2" x2="10.4" y2="16.8" stroke="#60a5fa" stroke-width="1.5" opacity=".65"/>
        <line x1="15.6" y1="5.2" x2="20.4" y2="9.8" stroke="#60a5fa" stroke-width="1.5" opacity=".65"/>
        <line x1="15.6" y1="16.8" x2="20.4" y2="12.2" stroke="#60a5fa" stroke-width="1.5" opacity=".65"/>
      </svg>
      <div>
        <div class="brand-name">Marcellus</div>
        <div class="brand-tag">IA local via Ollama</div>
      </div>
    </div>
    <button class="btn-new" id="newChat">+ Nova conversa</button>
    <button class="btn-folder" id="newFolderBtn">📁 Nova pasta</button>
    <input id="globalSearch" class="global-search" type="search"
           placeholder="🔍 Buscar em todas as conversas..." autocomplete="off">
    <div class="history" id="history"></div>
    <div class="status-bar" id="statusBar">Conectando...</div>
  </aside>

  <!-- Main -->
  <main class="main">
    <header class="topbar">
      <div class="topbar-title-area">
        <span class="topbar-title" id="chatTitle">Nova conversa</span>
        <input id="titleInput" style="display:none" placeholder="Nome da conversa">
        <button class="icon-btn" id="editTitleBtn" title="Renomear conversa" style="font-size:13px;width:26px;height:26px;flex-shrink:0">✏</button>
      </div>
      <button class="icon-btn" id="themeBtn" title="Alternar tema claro/escuro" style="font-size:15px; width:32px; height:32px; background:var(--panel-2); border:1px solid var(--line); border-radius:8px; cursor:pointer;">🌙</button>
      <button class="icon-btn" id="exportBtn" title="Exportar conversa como Markdown" style="font-size:15px; width:32px; height:32px; background:var(--panel-2); border:1px solid var(--line); border-radius:8px; cursor:pointer;">⬇</button>
      <div class="ctrl-row">
        <span class="ctrl-label">Modo</span>
        <select id="providerSelect">
          <option value="local">Local</option>
          <option value="cloud-direct">Cloud (API key)</option>
        </select>
        <span class="ctrl-label">Modelo</span>
        <span class="model-dot" id="modelDot"></span>
        <select id="modelSelect"></select>
        <input id="apiKey" type="password" placeholder="OLLAMA_API_KEY" style="display:none">
        <span class="ctrl-label">Personalidade</span>
        <select id="personalitySelect"></select>
        <label class="privacy-row" id="privacyLabel" style="display:none" title="Anonimiza dados antes de enviar">
          <input id="privacyMode" type="checkbox" checked> Privacidade
        </label>
        <label class="privacy-row" title="Confere a resposta contra as referências antes de entregar (mais lento; ideal para pareceres)">
          <input id="rigorMode" type="checkbox"> Rigoroso
        </label>
        <label class="privacy-row" title="Consulta a internet (LexML para legislação + busca geral) e injeta como referência">
          <input id="webMode" type="checkbox"> 🌐 Web
        </label>
      </div>
      <div class="workspace-row">
        <span class="ctrl-label">Workspace</span>
        <input id="workspacePath" placeholder="/caminho/da/pasta">
        <button class="workspace-btn" id="workspacePickBtn" title="Selecionar pasta">...</button>
        <button class="workspace-btn" id="workspaceSaveBtn" title="Vincular e indexar pasta">OK</button>
        <button class="workspace-btn" id="workspaceSyncBtn" title="Sincronizar arquivos">Sync</button>
        <button class="workspace-btn" id="workspaceClearBtn" title="Remover workspace">X</button>
        <span class="workspace-status" id="workspaceStatus">sem pasta</span>
      </div>
    </header>

    <section class="messages" id="messages">
      <div class="empty">Comece uma conversa. Anexos são guardados no histórico e textos entram como contexto.</div>
    </section>

    <div id="ctxBanner" style="display:none; padding: 0 16px;"></div>
    <footer class="composer">
      <form id="chatForm">
        <div class="composer-inner">
          <label class="file-label" title="Adicionar arquivos">
            ＋<input id="files" type="file" multiple>
          </label>
          <textarea id="prompt" rows="1" placeholder="Digite sua mensagem..."></textarea>
          <button class="btn-send" id="sendBtn" type="submit">Enviar</button>
        </div>
        <div class="file-list" id="fileList"></div>
      </form>
    </footer>
  </main>
</div>

<script>
  let conversations = [], folders = [], currentId = null, selectedFiles = [];
  let activeController = null, isGenerating = false;
  let localModels = [], cloudModels = [];
  let cloudConnected = false;
  let closedFolders = new Set(), menuConvId = null;

  const $ = id => document.getElementById(id);
  const themeBtn = $("themeBtn");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    themeBtn.textContent = t === "dark" ? "☀️" : "🌙";
    localStorage.setItem("marcellus_theme", t);
  }
  applyTheme(localStorage.getItem("marcellus_theme") || "light");
  themeBtn.onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

  const historyEl = $("history"), messagesEl = $("messages"), statusEl = $("statusBar");
  const chatTitleEl = $("chatTitle"), modelSelect = $("modelSelect");
  const personalitySelect = $("personalitySelect"), providerSelect = $("providerSelect");
  const apiKeyEl = $("apiKey"), privacyModeEl = $("privacyMode"), rigorModeEl = $("rigorMode"), webModeEl = $("webMode");
  const promptEl = $("prompt"), filesEl = $("files"), fileListEl = $("fileList");
  const sendBtn = $("sendBtn"), modelDot = $("modelDot");
  const titleInputEl = $("titleInput"), dropOverlay = $("dropOverlay");
  const convMenuEl = $("convMenu");
  const workspacePathEl = $("workspacePath"), workspaceStatusEl = $("workspaceStatus");

  // --- Utilities ---
  function fileIcon(name) {
    const ext = (name || "").split(".").pop().toLowerCase();
    return {pdf:"📄",docx:"📝",xlsx:"📊",xls:"📊",pptx:"📊",ppt:"📊",zip:"📦",py:"🐍",
            js:"📜",ts:"📜",html:"🌐",css:"🎨",json:"📋",md:"📋",
            csv:"📊",sh:"⚙",sql:"🗄",xml:"📋"}[ext] || "📄";
  }

  function renderAgentFiles(files) {
    if (!files?.length) return "";
    return '<div class="agent-files-row">' +
      files.map(f => {
        const isZip = (f.mime || "").includes("zip") || f.filename?.endsWith(".zip");
        const isPy = (f.filename || "").toLowerCase().endsWith(".py");
        const runBtn = isPy
          ? `<button class="exec-agent-btn" data-exec-file-id="${escapeHtml(f.id)}" title="Executar arquivo">&#9654;</button>`
          : '';
        let card;
        if (f.save_to_workspace) {
          card = `<a class="agent-file-card${isZip ? " zip" : ""}" href="/api/agent-files/${f.id}" data-agent-file-id="${escapeHtml(f.id)}" title="Salvar no workspace">
            <span class="file-icon">${fileIcon(f.filename)}</span>
            <span class="file-name">${escapeHtml(f.filename)}</span>
            <span class="file-dl">&#11015;</span>
          </a>`;
        } else {
          card = `<a class="agent-file-card${isZip ? " zip" : ""}" href="/api/agent-files/${f.id}" download="${escapeHtml(f.filename)}">
            <span class="file-icon">${fileIcon(f.filename)}</span>
            <span class="file-name">${escapeHtml(f.filename)}</span>
            <span class="file-dl">&#11015;</span>
          </a>`;
        }
        return `<div class="agent-file-wrap">${card}${runBtn}</div>`;
      }).join("") +
    "</div>";
  }

  messagesEl.addEventListener("click", async e => {
    const execBtn = e.target.closest("[data-exec-file-id]");
    if (execBtn) { e.preventDefault(); e.stopPropagation(); runAgentFile(execBtn); return; }
    const btn = e.target.closest("[data-agent-file-id]");
    if (!btn) return;
    e.preventDefault();
    const id = btn.dataset.agentFileId;
    btn.style.pointerEvents = "none";
    const old = btn.querySelector(".file-dl")?.textContent || "⬇";
    if (btn.querySelector(".file-dl")) btn.querySelector(".file-dl").textContent = "...";
    try {
      const data = await apiFetch(`/api/agent-files/${id}`, {method: "POST"});
      if (btn.querySelector(".file-dl")) btn.querySelector(".file-dl").textContent = old;
      statusEl.innerHTML = `<span class="mem-badge">${escapeHtml(data.message || "Arquivo salvo no workspace.")}</span>`;
    } catch (err) {
      if (btn.querySelector(".file-dl")) btn.querySelector(".file-dl").textContent = old;
      statusEl.textContent = err.message;
    } finally {
      btn.style.pointerEvents = "";
    }
  });

  function escapeHtml(v) {
    return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
                    .replaceAll('"',"&quot;").replaceAll("'","&#039;");
  }

  const execCodeStore = [];

  function renderContent(text) {
    const blocks = [];
    let t = String(text).replace(/```([\\w]*)\\n?([\\s\\S]*?)```/g, (_, lang, code) => {
      const i = blocks.length;
      const lbl = lang ? `<span class="code-lang">${escapeHtml(lang)}</span>` : '';
      const isPy = ['python', 'py', 'python3'].includes((lang || '').toLowerCase());
      const pre = `<pre class="code-block">${lbl}<code>${escapeHtml(code.trim())}</code></pre>`;
      if (isPy) {
        const idx = execCodeStore.length;
        execCodeStore.push(code.trim());
        const runBtn = `<button class="exec-btn" data-exec-idx="${idx}" onclick="runCode(this)">&#9654; Executar</button>`;
        blocks.push(`<div class="code-wrap">${pre}${runBtn}</div>`);
      } else {
        blocks.push(pre);
      }
      return `\\uE000${i}\\uE001`;
    });
    t = t.replace(/`([^`\\n]{1,300})`/g, (_, c) => {
      const i = blocks.length;
      blocks.push(`<code class="inline-code">${escapeHtml(c)}</code>`);
      return `\\uE000${i}\\uE001`;
    });
    t = escapeHtml(t).replace(/\\n/g, "<br>");
    return t.replace(/\\uE000(\\d+)\\uE001/g, (_, i) => blocks[+i]);
  }

  function modelStorageKey() {
    return providerSelect.value === "cloud-direct" ? "nx_cloud_model" : "nx_local_model";
  }

  function setGenerating(v) {
    isGenerating = v;
    sendBtn.textContent = v ? "Stop" : "Enviar";
    sendBtn.classList.toggle("stop", v);
    sendBtn.type = v ? "button" : "submit";
  }

  // --- Model dot status ---
  async function refreshModelDot(updateStatus = true) {
    const isCloud = providerSelect.value === "cloud-direct";
    if (isCloud) {
      const active = cloudConnected && !!modelSelect.value;
      modelDot.className = "model-dot" + (active ? " loaded" : "");
      modelDot.title = active ? "Conectado ao Ollama Cloud" : "Cloud não conectado (informe a API key)";
      return;
    }
    try {
      const data = await apiFetch("/api/status");
      const loaded = data.loaded_models || [];
      const active = loaded.some(m => m === modelSelect.value);
      modelDot.className = "model-dot" + (active ? " loaded" : "");
      modelDot.title = active ? "Modelo carregado na RAM" : "Modelo não carregado (1ª resposta será mais lenta)";
      if (updateStatus) {
        statusEl.innerHTML = `<span class="mem-badge">💾 ${data.memory_count} mem · ${data.insight_count} insights</span>`;
      }
    } catch (_) {}
  }

  // --- API ---
  async function apiFetch(path, opts = {}) {
    const r = await fetch(path, opts);
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || "Erro na requisição.");
    return d;
  }

  // --- Models ---
  async function loadModels() {
    try {
      const data = await apiFetch("/api/models");
      localModels = data.models || [];
      renderModelOptions();
      if (providerSelect.value === "local" && !modelSelect.value && data.default_model) {
        modelSelect.value = data.default_model;
        localStorage.setItem(modelStorageKey(), data.default_model);
      }
      refreshModelDot();
    } catch (_) {
      statusEl.textContent = "Ollama indisponível. Rode: ollama serve";
    }
  }

  async function loadCloudModels() {
    const key = apiKeyEl.value.trim();
    if (!key) { cloudModels = []; cloudConnected = false; renderModelOptions(); refreshModelDot(); return; }
    try {
      const data = await apiFetch("/api/cloud-models", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({api_key: key}),
      });
      cloudModels = data.models || [];
      cloudConnected = true;
      renderModelOptions();
    } catch (e) {
      cloudConnected = false;
      statusEl.textContent = e.message;
    }
    refreshModelDot();
  }

  function renderModelOptions() {
    const isCloud = providerSelect.value !== "local";
    const models = isCloud ? cloudModels : localModels;
    const last = localStorage.getItem(modelStorageKey()) || "";
    modelSelect.innerHTML = '<option value="">Selecione o modelo</option>';
    models.forEach(m => {
      const o = document.createElement("option");
      o.value = o.textContent = m;
      modelSelect.appendChild(o);
    });
    modelSelect.value = models.includes(last) ? last : "";
  }

  // --- Personalities ---
  async function loadPersonalities() {
    const data = await apiFetch("/api/personalities");
    personalitySelect.innerHTML = "";
    (data.personalities || []).forEach(p => {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = `${p.emoji} ${p.name}`;
      o.title = p.description;
      personalitySelect.appendChild(o);
    });
  }

  // --- Conversations ---
  function makeConvEl(c) {
    const row = document.createElement("div");
    row.className = "conv" + (c.id === currentId ? " active" : "");
    const emoji = personalitySelect.querySelector(`option[value="${c.personality}"]`)?.textContent?.split(" ")[0] || "🤖";
    row.innerHTML = `
      <div class="conv-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</div>
      <span class="conv-personality" title="${c.personality || 'atlas'}">${emoji}</span>
      <button class="icon-btn" title="Opções" style="font-size:15px">⋯</button>`;
    row.querySelector(".conv-title").onclick = () => loadConversation(c.id);
    row.querySelector(".icon-btn").onclick = (e) => {
      e.stopPropagation();
      openConvMenu(c.id, e.currentTarget);
    };
    return row;
  }

  function makeFolderSection(f, convs) {
    const isOpen = !closedFolders.has(f.id);
    const sec = document.createElement("div");
    sec.className = "folder-section";
    sec.innerHTML = `
      <div class="folder-header">
        <span class="folder-arrow${isOpen ? " open" : ""}">▶</span>
        <span class="folder-name-text" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
        <button class="icon-btn" title="Renomear" style="font-size:11px;width:22px;height:22px">✏</button>
        <button class="icon-btn danger" title="Excluir pasta" style="font-size:11px;width:22px;height:22px">✕</button>
      </div>
      <div class="folder-body${isOpen ? "" : " collapsed"}"></div>`;
    const body = sec.querySelector(".folder-body");
    convs.forEach(c => body.appendChild(makeConvEl(c)));
    const header = sec.querySelector(".folder-header");
    header.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      const arrow = sec.querySelector(".folder-arrow");
      const nowOpen = arrow.classList.toggle("open");
      body.classList.toggle("collapsed", !nowOpen);
      if (nowOpen) closedFolders.delete(f.id);
      else closedFolders.add(f.id);
    });
    sec.querySelectorAll("button")[0].onclick = async (e) => {
      e.stopPropagation();
      const name = prompt("Novo nome da pasta:", f.name);
      if (!name?.trim()) return;
      await apiFetch(`/api/folders/${f.id}`, {
        method: "PATCH", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: name.trim()}),
      });
      folders = folders.map(x => x.id === f.id ? {...x, name: name.trim()} : x);
      renderHistory();
    };
    sec.querySelectorAll("button")[1].onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Excluir pasta "${f.name}"? As conversas voltam para sem pasta.`)) return;
      await apiFetch(`/api/folders/${f.id}`, {method: "DELETE"});
      closedFolders.delete(f.id);
      conversations = conversations.map(c => c.folder_id === f.id ? {...c, folder_id: null} : c);
      folders = folders.filter(x => x.id !== f.id);
      renderHistory();
    };
    return sec;
  }

  const globalSearchEl = $("globalSearch");
  let searchTimer = null;
  globalSearchEl.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = globalSearchEl.value.trim();
    if (q.length < 3) { renderHistory(); return; }
    searchTimer = setTimeout(async () => {
      const data = await apiFetch(`/api/search?q=${encodeURIComponent(q)}`);
      historyEl.innerHTML = "";
      if (!data.results.length) {
        historyEl.innerHTML = '<div class="empty">Nada encontrado.</div>';
        return;
      }
      data.results.forEach(r => {
        const div = document.createElement("div");
        div.className = "search-hit";
        div.innerHTML = `<div class="sh-title">${escapeHtml(r.title)} <span style="font-weight:400;color:var(--muted)">· ${r.when}</span></div>` +
                        `<div class="sh-snippet">${escapeHtml(r.snippet)}</div>`;
        div.onclick = () => { globalSearchEl.value = ""; loadConversation(r.conversation_id); };
        historyEl.appendChild(div);
      });
    }, 300);
  });

  function renderHistory() {
    historyEl.innerHTML = "";
    const byFolder = {};
    const unfiled = [];
    conversations.forEach(c => {
      if (c.folder_id) (byFolder[c.folder_id] = byFolder[c.folder_id] || []).push(c);
      else unfiled.push(c);
    });
    unfiled.forEach(c => historyEl.appendChild(makeConvEl(c)));
    folders.forEach(f => historyEl.appendChild(makeFolderSection(f, byFolder[f.id] || [])));
  }

  function renderWorkspaceSummary(summary = {}) {
    const path = summary.workspace_path || "";
    workspacePathEl.value = path;
    if (!path) {
      workspaceStatusEl.textContent = "sem pasta";
      workspaceStatusEl.title = "";
      return;
    }
    const indexed = summary.indexed_files || 0;
    const skipped = summary.skipped_files || 0;
    workspaceStatusEl.textContent = `${indexed} arquivos${skipped ? ` · ${skipped} sem texto` : ""}`;
    workspaceStatusEl.title = path;
  }

  async function saveWorkspacePath() {
    if (!currentId) await createConversation();
    const path = workspacePathEl.value.trim();
    workspaceStatusEl.textContent = path ? "indexando..." : "removendo...";
    const data = await apiFetch(`/api/conversations/${currentId}/workspace`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path}),
    });
    renderWorkspaceSummary(data.workspace);
    conversations = data.conversations || conversations;
    renderHistory();
    const s = data.workspace || {};
    if (path) statusEl.innerHTML = `<span class="mem-badge">Workspace: +${s.added || 0} · alt ${s.updated || 0} · rem ${s.removed || 0}</span>`;
  }

  async function pickWorkspacePath() {
    workspaceStatusEl.textContent = "selecionando...";
    try {
      const data = await apiFetch("/api/select-folder", {method: "POST"});
      if (!data.path) {
        renderWorkspaceSummary({workspace_path: workspacePathEl.value});
        return;
      }
      workspacePathEl.value = data.path;
      await saveWorkspacePath();
    } catch (err) {
      workspaceStatusEl.textContent = err.message;
    }
  }

  async function syncWorkspace() {
    if (!currentId) return;
    workspaceStatusEl.textContent = "sincronizando...";
    const data = await apiFetch(`/api/conversations/${currentId}/workspace/sync`, {method: "POST"});
    renderWorkspaceSummary(data.workspace);
    const s = data.workspace || {};
    statusEl.innerHTML = `<span class="mem-badge">Workspace: +${s.added || 0} · alt ${s.updated || 0} · rem ${s.removed || 0}</span>`;
  }

  async function clearWorkspace() {
    if (!currentId) return;
    workspacePathEl.value = "";
    await saveWorkspacePath();
  }

  // --- Conv context menu ---
  function openConvMenu(convId, triggerEl) {
    menuConvId = convId;
    const c = conversations.find(x => x.id === convId);
    const sel = $("convFolderSel");
    sel.innerHTML = '<option value="">Sem pasta</option>' +
      folders.map(f => `<option value="${f.id}"${c?.folder_id === f.id ? " selected" : ""}>${escapeHtml(f.name)}</option>`).join("");
    const rect = triggerEl.getBoundingClientRect();
    convMenuEl.style.top = (rect.bottom + 4) + "px";
    convMenuEl.style.left = Math.max(4, rect.right - 215) + "px";
    convMenuEl.classList.add("open");
  }

  document.addEventListener("click", (e) => {
    if (!convMenuEl.contains(e.target)) convMenuEl.classList.remove("open");
  });
  convMenuEl.addEventListener("click", e => e.stopPropagation());

  $("convFolderSel").addEventListener("change", async () => {
    if (!menuConvId) return;
    const folderId = $("convFolderSel").value || null;
    await apiFetch(`/api/conversations/${menuConvId}`, {
      method: "PATCH", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({folder_id: folderId}),
    });
    conversations = conversations.map(c => c.id === menuConvId ? {...c, folder_id: folderId} : c);
    convMenuEl.classList.remove("open");
    menuConvId = null;
    renderHistory();
  });

  $("convDeleteBtn").addEventListener("click", () => {
    if (!menuConvId) return;
    const id = menuConvId;
    convMenuEl.classList.remove("open");
    menuConvId = null;
    deleteConversation(id);
  });

  function renderMessages(messages) {
    messagesEl.innerHTML = "";
    if (!messages.length) {
      messagesEl.innerHTML = '<div class="empty">Nenhuma mensagem ainda. Comece a conversar!</div>';
      return;
    }
    messages.forEach(m => appendRenderedMessage(m.role, m.content, m.attachments || [], m.agent_files || [], m.id, m.feedback || ""));
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendRenderedMessage(role, content, attachments = [], agentFiles = [], msgId = "", feedback = "") {
    const empty = messagesEl.querySelector(".empty");
    if (empty) empty.remove();
    const div = document.createElement("div");
    div.className = `message ${role}`;
    const label = role === "user" ? "Você" : "Assistente";
    let atts = "";
    if (attachments.length) {
      atts = '<div class="attachments">' + attachments.map(f =>
        f.id
          ? `<a class="pill" href="/uploads/${f.id}" target="_blank">${escapeHtml(f.name)}</a>`
          : `<span class="pill">${escapeHtml(f.name)}</span>`
      ).join("") + "</div>";
    }
    let fb = "";
    if (role === "assistant" && msgId) {
      fb = `<div class="feedback" data-mid="${msgId}">
        <button class="fb-btn${feedback === "up" ? " active" : ""}" data-v="up" title="Resposta boa — vira exemplo de treino">👍</button>
        <button class="fb-btn${feedback === "down" ? " active" : ""}" data-v="down" title="Resposta ruim — vira exemplo negativo">👎</button>
      </div>`;
    }
    div.innerHTML = `<span class="meta">${label}</span>${renderContent(content || "")}${atts}${renderAgentFiles(agentFiles)}${fb}`;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  messagesEl.addEventListener("click", async (e) => {
    const btn = e.target.closest(".fb-btn");
    if (!btn) return;
    const bar = btn.closest(".feedback");
    const value = btn.classList.contains("active") ? "" : btn.dataset.v;
    try {
      await apiFetch(`/api/messages/${bar.dataset.mid}/feedback`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({value}),
      });
      bar.querySelectorAll(".fb-btn").forEach(b => b.classList.remove("active"));
      if (value) btn.classList.add("active");
    } catch (err) { /* feedback é best-effort; não interrompe o chat */ }
  });

  async function loadConversations() {
    const [convData, folderData] = await Promise.all([
      apiFetch("/api/conversations"),
      apiFetch("/api/folders"),
    ]);
    conversations = convData.conversations;
    folders = folderData.folders;
    if (!currentId && conversations.length) currentId = conversations[0].id;
    renderHistory();
    if (currentId) await loadConversation(currentId);
    else { chatTitleEl.textContent = "Nova conversa"; renderWorkspaceSummary({}); renderMessages([]); }
  }

  async function loadConversation(id) {
    const data = await apiFetch(`/api/conversations/${id}`);
    currentId = id;
    chatTitleEl.textContent = data.conversation.title;
    const p = data.conversation.personality || "atlas";
    if (personalitySelect.querySelector(`option[value="${p}"]`)) personalitySelect.value = p;
    renderWorkspaceSummary(data.workspace || data.conversation);
    renderHistory();
    renderMessages(data.messages);
    $("ctxBanner").style.display = "none";
  }

  async function createConversation() {
    const data = await apiFetch("/api/conversations", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({personality: personalitySelect.value || "atlas"}),
    });
    currentId = data.conversation.id;
    await loadConversations();
  }

  async function deleteConversation(id) {
    if (!confirm("Excluir esta conversa?")) return;
    await apiFetch(`/api/conversations/${id}`, {method: "DELETE"});
    if (currentId === id) currentId = null;
    await loadConversations();
  }

  // --- Personality change ---
  personalitySelect.addEventListener("change", async () => {
    if (!currentId) return;
    await apiFetch(`/api/conversations/${currentId}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({personality: personalitySelect.value}),
    });
    renderHistory();
  });

  // --- Provider ---
  function syncProvider() {
    const isCloud = providerSelect.value === "cloud-direct";
    apiKeyEl.style.display = isCloud ? "" : "none";
    $("privacyLabel").style.display = isCloud ? "" : "none";
    renderModelOptions();
    if (isCloud) loadCloudModels();
  }

  providerSelect.addEventListener("change", syncProvider);
  apiKeyEl.addEventListener("change", () => {
    const k = apiKeyEl.value.trim();
    k ? localStorage.setItem("nx_api_key", k) : localStorage.removeItem("nx_api_key");
    loadCloudModels();
  });
  modelSelect.addEventListener("change", () => {
    if (modelSelect.value) localStorage.setItem(modelStorageKey(), modelSelect.value);
    refreshModelDot();
  });

  // --- Files & drag-drop ---
  function updateFileList() {
    fileListEl.innerHTML = selectedFiles.map(f =>
      `<span class="pill">${escapeHtml(f.name)} (${Math.ceil(f.size/1024)} KB)</span>`
    ).join("");
  }

  filesEl.addEventListener("change", () => {
    selectedFiles = selectedFiles.concat(Array.from(filesEl.files));
    filesEl.value = "";
    updateFileList();
  });

  let dragDepth = 0;
  document.addEventListener("dragenter", (e) => {
    if (![...e.dataTransfer.items].some(i => i.kind === "file")) return;
    dragDepth++;
    dropOverlay.classList.add("active");
  });
  document.addEventListener("dragleave", () => {
    if (--dragDepth <= 0) { dragDepth = 0; dropOverlay.classList.remove("active"); }
  });
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropOverlay.classList.remove("active");
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) { selectedFiles = selectedFiles.concat(files); updateFileList(); }
  });

  // --- Textarea auto-resize ---
  promptEl.addEventListener("input", () => {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 150) + "px";
  });
  promptEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chatForm").requestSubmit(); }
  });

  // --- Title edit ---
  function startTitleEdit() {
    if (!currentId) return;
    titleInputEl.value = chatTitleEl.textContent;
    chatTitleEl.style.display = "none";
    titleInputEl.style.display = "";
    titleInputEl.focus();
    titleInputEl.select();
  }
  async function saveTitleEdit() {
    const name = titleInputEl.value.trim();
    chatTitleEl.style.display = "";
    titleInputEl.style.display = "none";
    if (!name || name === chatTitleEl.textContent || !currentId) return;
    chatTitleEl.textContent = name;
    await apiFetch(`/api/conversations/${currentId}`, {
      method: "PATCH", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title: name}),
    });
    conversations = conversations.map(c => c.id === currentId ? {...c, title: name} : c);
    renderHistory();
  }
  $("editTitleBtn").addEventListener("click", startTitleEdit);
  chatTitleEl.addEventListener("dblclick", startTitleEdit);
  titleInputEl.addEventListener("blur", saveTitleEdit);
  titleInputEl.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); saveTitleEdit(); }
    if (e.key === "Escape") { chatTitleEl.style.display = ""; titleInputEl.style.display = "none"; }
  });

  // --- New folder ---
  $("newFolderBtn").addEventListener("click", async () => {
    const name = prompt("Nome da pasta:");
    if (!name?.trim()) return;
    const data = await apiFetch("/api/folders", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: name.trim()}),
    });
    folders = data.folders;
    renderHistory();
  });

  $("newChat").addEventListener("click", createConversation);

  $("exportBtn").addEventListener("click", () => {
    if (currentId) window.open(`/api/conversations/${currentId}/export`, "_blank");
  });

  $("workspacePickBtn").addEventListener("click", pickWorkspacePath);
  $("workspaceSaveBtn").addEventListener("click", saveWorkspacePath);
  $("workspaceSyncBtn").addEventListener("click", syncWorkspace);
  $("workspaceClearBtn").addEventListener("click", clearWorkspace);
  workspacePathEl.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); saveWorkspacePath(); }
  });

  // --- Stop ---
  sendBtn.addEventListener("click", () => {
    if (!isGenerating || !activeController) return;
    activeController.abort();
    activeController = null;
    setGenerating(false);
    statusEl.textContent = "Cancelado.";
  });

  // --- Submit ---
  $("chatForm").addEventListener("submit", async e => {
    e.preventDefault();
    if (isGenerating) return;
    const prompt = promptEl.value.trim();
    if (!prompt && !selectedFiles.length) return;
    if (!currentId) await createConversation();

    const form = new FormData();
    form.append("prompt", prompt);
    form.append("model", modelSelect.value);
    form.append("provider", providerSelect.value);
    form.append("api_key", apiKeyEl.value.trim());
    form.append("privacy_mode", privacyModeEl.checked ? "1" : "0");
    form.append("rigor_mode", rigorModeEl.checked ? "1" : "0");
    form.append("web_mode", webModeEl.checked ? "1" : "0");
    form.append("personality", personalitySelect.value || "atlas");
    selectedFiles.forEach(f => form.append("files", f));

    const optFiles = selectedFiles.map(f => ({name: f.name}));
    appendRenderedMessage("user", prompt || "[arquivos anexados]", optFiles);
    const waitEl = appendRenderedMessage("assistant", "");
    waitEl.querySelector(".meta").insertAdjacentHTML("afterend", '<span class="thinking">Gerando resposta...</span>');

    promptEl.value = ""; promptEl.style.height = "auto";
    selectedFiles = []; fileListEl.innerHTML = "";
    setGenerating(true);
    activeController = new AbortController();

    let firstChunk = false;
    const wasModelLoaded = modelDot.classList.contains("loaded");
    const loadTimer = setTimeout(() => {
      if (!firstChunk) {
        const thinkingText = wasModelLoaded
          ? "Processando resposta, aguarde..."
          : "Carregando modelo na memória, aguarde...";
        waitEl.innerHTML = `<span class="meta">Assistente</span><span class="thinking">${thinkingText}</span>`;
        if (!wasModelLoaded) modelDot.className = "model-dot";
      }
    }, 8000);

    try {
      const resp = await fetch(`/api/conversations/${currentId}/chat`, {
        method: "POST", body: form, signal: activeController.signal,
      });
      if (!resp.ok || !resp.body) {
        const d = await resp.json().catch(() => ({}));
        throw new Error(d.error || "Erro na requisição.");
      }

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "", assistantText = "";

      while (true) {
        let result;
        try { result = await reader.read(); }
        catch (_) { throw new Error("Conexão com o modelo interrompida. Verifique se o Ollama está rodando."); }

        const { value, done } = result;
        if (done) break;

        buf += dec.decode(value, {stream: true});
        const lines = buf.split("\\n");
        buf = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          let ev;
          try { ev = JSON.parse(line); }
          catch (_) { continue; }

          if (ev.type === "meta") {
            conversations = ev.conversations;
            chatTitleEl.textContent = ev.conversation.title;
            renderHistory();
          }
          if (ev.type === "chunk") {
            firstChunk = true;
            clearTimeout(loadTimer);
            assistantText += ev.content;
            waitEl.innerHTML = `<span class="meta">Assistente</span>${renderContent(assistantText)}`;
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
          if (ev.type === "status") {
            statusEl.innerHTML = `<span class="mem-badge">${escapeHtml(ev.message || "")}</span>`;
          }
          if (ev.type === "done") {
            conversations = ev.conversations;
            chatTitleEl.textContent = ev.conversation.title;
            renderHistory();
            renderMessages(ev.messages);
            const mt = ev.metrics || {};
            const perf = mt.response_seconds
              ? ` · ${mt.response_seconds}s${mt.first_token_seconds ? ` · 1º token ${mt.first_token_seconds}s` : ""}`
              : "";
            const ctx = mt.prompt_tokens_estimate ? ` · ctx ~${mt.prompt_tokens_estimate} tok` : "";
            const rag = mt.attachment_chunks
              ? ` · docs ${mt.attachment_chunks_used || 0}/${mt.attachment_chunks}`
              : "";
            const wsChanged = (mt.workspace_added || 0) + (mt.workspace_updated || 0) + (mt.workspace_removed || 0);
            const ws = wsChanged ? ` · ws +${mt.workspace_added || 0}/alt ${mt.workspace_updated || 0}/rem ${mt.workspace_removed || 0}` : "";
            const reason = mt.reasoning ? ` · reasoning: ${mt.reasoning}` : "";
            const rig = mt.rigor ? " · ✓ verificado" : "";
            const gl = ev.global_count != null ? ` · ${ev.global_count} global` : "";
            const memInfo = `💾 ${ev.memory_count} mem · ${ev.insight_count} insights${gl} · ${ev.memories_used || 0} usadas${perf}${ctx}${rag}${ws}${reason}${rig}`;
            statusEl.innerHTML = `<span class="mem-badge">${memInfo}</span>`;

            const pct = mt.context_pct || 0;
            const ctxBanner = $("ctxBanner");
            if (pct >= 85) {
              ctxBanner.style.display = "block";
              ctxBanner.innerHTML = `<div class="ctx-banner alert">
                <span>⚠️ Contexto ${pct}% cheio — a conversa pode começar a perder coerência.</span>
                <div class="ctx-bar"><div class="ctx-bar-fill" style="width:${pct}%"></div></div>
                <a onclick="$('newChat').click()">Iniciar nova conversa</a>
              </div>`;
            } else if (pct >= 70) {
              ctxBanner.style.display = "block";
              ctxBanner.innerHTML = `<div class="ctx-banner warn">
                <span>Contexto ${pct}% utilizado</span>
                <div class="ctx-bar"><div class="ctx-bar-fill" style="width:${pct}%"></div></div>
                <a onclick="$('newChat').click()">Nova conversa</a>
              </div>`;
            } else {
              ctxBanner.style.display = "none";
            }

            if (providerSelect.value === "cloud-direct") cloudConnected = true;
            refreshModelDot(false);
          }
          if (ev.type === "error") throw new Error(ev.error);
        }
      }
    } catch (err) {
      clearTimeout(loadTimer);
      if (err.name === "AbortError") {
        waitEl.remove();
        appendRenderedMessage("assistant", "Solicitação cancelada.");
        return;
      }
      waitEl.remove();
      appendRenderedMessage("assistant", err.message);
      statusEl.textContent = err.message;
    } finally {
      clearTimeout(loadTimer);
      activeController = null;
      setGenerating(false);
    }
  });

  // --- Code execution ---
  async function runCode(btn) {
    const idx = parseInt(btn.dataset.execIdx, 10);
    const code = execCodeStore[idx];
    if (code === undefined) return;
    btn.disabled = true;
    btn.textContent = '\\u23F3 Executando...';
    const prevOut = btn.nextElementSibling;
    if (prevOut?.classList.contains('exec-output')) prevOut.remove();
    try {
      const data = await apiFetch('/api/execute-code', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({code}),
      });
      const out = document.createElement('div');
      out.className = 'exec-output' + (data.ok ? '' : ' error');
      let html = '';
      if (data.stdout) html += escapeHtml(data.stdout).replace(/\\n/g, '<br>');
      if (data.stderr) {
        if (html) html += '<br>';
        html += '<span style="color:#fca5a5">' + escapeHtml(data.stderr).replace(/\\n/g, '<br>') + '</span>';
      }
      if (!html) html = data.ok
        ? '<span style="color:#94a3b8">(sem output)</span>'
        : '<span style="color:#fca5a5">(erro desconhecido)</span>';
      (data.images || []).forEach(b64 => {
        html += '<img src="data:image/png;base64,' + b64 + '" alt="Gr\\u00e1fico gerado" style="max-width:100%;border-radius:6px;margin-top:8px;display:block">';
      });
      out.innerHTML = html;
      const outputText = [data.stdout, data.stderr].filter(Boolean).join('\\n').trim();
      if (outputText || (data.images || []).length) {
        const sb = document.createElement('button');
        sb.className = 'exec-send-btn';
        sb.textContent = '\\u2191 Enviar resultado para o chat';
        sb.onclick = () => sendCodeOutput(outputText, data.images || []);
        out.appendChild(sb);
      }
      btn.after(out);
    } catch (err) {
      const out = document.createElement('div');
      out.className = 'exec-output error';
      out.textContent = err.message;
      btn.after(out);
    } finally {
      btn.disabled = false;
      btn.textContent = '\\u25b6 Executar';
    }
  }

  async function runAgentFile(btn) {
    const fileId = btn.dataset.execFileId;
    btn.disabled = true;
    btn.textContent = '\\u23F3';
    const wrap = btn.closest('.agent-file-wrap');
    const prevOut = wrap?.nextElementSibling;
    if (prevOut?.classList.contains('exec-output')) prevOut.remove();
    try {
      const resp = await fetch('/api/agent-files/' + fileId);
      if (!resp.ok) throw new Error('Erro ao carregar arquivo.');
      const code = await resp.text();
      const data = await apiFetch('/api/execute-code', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({code}),
      });
      const out = document.createElement('div');
      out.className = 'exec-output' + (data.ok ? '' : ' error');
      let html = '';
      if (data.stdout) html += escapeHtml(data.stdout).replace(/\\n/g, '<br>');
      if (data.stderr) {
        if (html) html += '<br>';
        html += '<span style="color:#fca5a5">' + escapeHtml(data.stderr).replace(/\\n/g, '<br>') + '</span>';
      }
      if (!html) html = data.ok
        ? '<span style="color:#94a3b8">(sem output)</span>'
        : '<span style="color:#fca5a5">(erro desconhecido)</span>';
      (data.images || []).forEach(b64 => {
        html += '<img src="data:image/png;base64,' + b64 + '" style="max-width:100%;border-radius:6px;margin-top:8px;display:block">';
      });
      out.innerHTML = html;
      const outputText = [data.stdout, data.stderr].filter(Boolean).join('\\n').trim();
      if (outputText || (data.images || []).length) {
        const sb = document.createElement('button');
        sb.className = 'exec-send-btn';
        sb.textContent = '\\u2191 Enviar resultado para o chat';
        sb.onclick = () => sendCodeOutput(outputText, data.images || []);
        out.appendChild(sb);
      }
      (wrap || btn).after(out);
    } catch (err) {
      const out = document.createElement('div');
      out.className = 'exec-output error';
      out.textContent = err.message;
      (wrap || btn).after(out);
    } finally {
      btn.disabled = false;
      btn.textContent = '\\u25b6';
    }
  }

  function sendCodeOutput(output, images) {
    let msg = '';
    if (output) msg = 'Output do c\\u00f3digo:\\n```\\n' + output + '\\n```';
    if (images.length) msg += (msg ? '\\n\\n' : '') + images.length + ' imagem(ns) gerada(s) \\u2014 vis\\u00edvel acima no chat';
    promptEl.value = msg;
    promptEl.style.height = 'auto';
    promptEl.style.height = Math.min(promptEl.scrollHeight, 150) + 'px';
    promptEl.focus();
  }

  // --- Init ---
  apiKeyEl.value = localStorage.getItem("nx_api_key") || "";
  loadPersonalities().then(() => {
    syncProvider();
    loadModels();
    loadConversations();
  });
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_attachment_context(
    query: str,
    attachments: list[dict[str, Any]],
    provider: str = "local",
    conv_id: str = "",
) -> tuple[str, int, int]:
    budget = CLOUD_ATTACHMENT_CHARS if provider == "cloud-direct" else LOCAL_ATTACHMENT_CHARS
    query_tokens = tokenize(query)
    broad_query = not query_tokens or any(
        term in query.lower()
        for term in ("resum", "sumar", "todo", "inteiro", "arquivo", "documento", "geral", "tudo", "complet")
    )
    retrieval_limit = CLOUD_RETRIEVAL_LIMIT if provider == "cloud-direct" else LOCAL_RETRIEVAL_LIMIT

    # Primary: use persistent chunk index (covers entire conversation's documents)
    if conv_id:
        chunks = database.search_doc_chunks(conv_id, query_tokens, retrieval_limit, budget, broad_query, query_text=query)
        if chunks:
            total = database.count_doc_chunks(conv_id)
            parts = [
                f"Arquivo: {str(c.get('filename') or c.get('attachment_id', ''))[:60]} "
                f"| trecho {int(c['chunk_index']) + 1}\n{c['content']}"
                for c in chunks
            ]
            context = (
                "Trechos relevantes dos arquivos. Use-os para responder de forma completa "
                "à pergunta atual:\n" + "\n\n".join(parts)
            )
            return context, total, len(chunks)

    # Fallback: re-chunk from text_preview (no persistent index yet or no conv_id)
    text_files = [
        (str(f.get("name", "arquivo")), str(f.get("text_preview", "")).strip())
        for f in attachments if f.get("text_preview")
    ]
    if not text_files:
        if attachments:
            names = ", ".join(str(f.get("name", "arquivo")) for f in attachments)
            return f"Arquivos sem texto legível: {names}", len(attachments), 0
        return "", 0, 0

    ranked: list[tuple[float, int, str, str]] = []
    total_chunks = 0
    for name, text in text_files:
        file_chunks = mem.chunk_text(text)
        total_chunks += len(file_chunks)
        for index, chunk in enumerate(file_chunks):
            score = 1.0 / (index + 1) if broad_query else similarity(query_tokens, tokenize(chunk))
            if score > 0:
                ranked.append((score, index, name, chunk))

    if not ranked:
        for name, text in text_files:
            for index, chunk in enumerate(mem.chunk_text(text)[:2]):
                ranked.append((1.0 / (index + 1), index, name, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    used_chars = 0
    for _, index, name, chunk in ranked:
        block = f"Arquivo: {name} | trecho {index + 1}\n{chunk}"
        if selected and used_chars + len(block) > budget:
            continue
        selected.append(block)
        used_chars += len(block)
        if used_chars >= budget:
            break

    if not selected:
        return "", total_chunks, 0
    context = (
        "Trechos relevantes dos arquivos anexados. Use-os para responder de forma completa "
        "à pergunta atual:\n" + "\n\n".join(selected)
    )
    return context, total_chunks, len(selected)


def build_user_prompt(
    user_prompt: str,
    attachments: list[dict[str, Any]],
    provider: str = "local",
    conv_id: str = "",
) -> tuple[str, dict[str, int]]:
    attachment_context, total_chunks, selected_chunks = build_attachment_context(
        user_prompt, attachments, provider, conv_id,
    )
    parts = [user_prompt.strip()]
    if attachment_context:
        parts.append(attachment_context)
    full_prompt = "\n\n".join(p for p in parts if p)
    return full_prompt, {
        "attachment_count": len(attachments),
        "attachment_chunks": total_chunks,
        "attachment_chunks_used": selected_chunks,
    }


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(mem.estimate_tokens(m.get("content", "")) for m in messages)


def build_ollama_messages(
    conv_id: str,
    user_prompt: str,
    memories: list[dict[str, Any]],
    personality_id: str = "atlas",
    anonymize: bool = False,
    provider: str = "local",
    raw_prompt: str = "",
    knowledge: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    p = get_personality(personality_id)
    is_cloud = provider == "cloud-direct"
    memory_budget = (
        CLOUD_MEMORY_CONTEXT_TOKEN_BUDGET if is_cloud else MEMORY_CONTEXT_TOKEN_BUDGET
    )
    history_budget = (
        CLOUD_HISTORY_CONTEXT_TOKEN_BUDGET if is_cloud else HISTORY_CONTEXT_TOKEN_BUDGET
    )
    history_window = 24 if is_cloud else 12

    # Separate global (cross-conversation) knowledge from conversation memories
    global_lines: list[str] = []
    conv_lines: list[str] = []
    mem_tokens = 0

    for m in memories:
        content = str(m["content"]).strip()
        if not content:
            continue
        role = str(m.get("source_role", ""))
        if role == "global_insight":
            line = f"- {content}"
            dest = global_lines
        elif role == "insight":
            line = f"- {content}"
            dest = conv_lines
        else:
            line = f"- {content}"
            dest = conv_lines
        line_tokens = mem.estimate_tokens(line)
        if mem_tokens + line_tokens > memory_budget:
            continue
        dest.append(line)
        mem_tokens += line_tokens

    global_text = "\n".join(global_lines)
    conv_text = "\n".join(conv_lines)

    if anonymize:
        user_prompt = anonymize_for_cloud(user_prompt)
        global_text = anonymize_for_cloud(global_text)
        conv_text = anonymize_for_cloud(conv_text)

    # System prompt em camadas: seções rotuladas evitam que modelos pequenos
    # misturem identidade, fatos permanentes, instruções e exemplos
    system_parts = [f"## IDENTIDADE\n{p.system_prompt}"]
    facts_block = facts.get_facts_block()
    if facts_block:
        system_parts.append(
            "## SOBRE O USUÁRIO (conhecimento de fundo)\n"
            "Incorpore com naturalidade, como quem simplesmente conhece a pessoa. "
            "NUNCA mencione esta seção, não recite estes itens em lista e não cite "
            "nomes de sistemas internos (embeddings, RAG, memórias) sem necessidade. "
            "Estes são os ÚNICOS fatos conhecidos sobre o usuário: não deduza nem "
            "invente outros (aniversário, família, gostos, eventos) — o que não "
            "estiver aqui ou na conversa, você simplesmente não sabe.\n"
            + facts_block
        )
    system_parts.append(f"## DATA ATUAL\n{time.strftime('%d/%m/%Y')}")
    system = "\n\n".join(system_parts)
    system += few_shot.build_few_shot_block(
        is_cloud=is_cloud, personality_id=p.id, prompt=raw_prompt or user_prompt,
    )
    if anonymize:
        system += " Textos podem estar anonimizados; respeite os marcadores."
    if file_gen.wants_file_creation(user_prompt):
        system += file_gen.FILE_CREATION_HINT

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    history = database.get_messages(conv_id)[-history_window:-1]
    while history and history[-1]["role"] == "user":
        history.pop()

    selected_history: list[dict[str, Any]] = []
    history_tokens = 0
    for item in reversed(history):
        if item["role"] in {"user", "assistant"}:
            content = str(item["content"])
            item_tokens = mem.estimate_tokens(content)
            if selected_history and history_tokens + item_tokens > history_budget:
                continue
            selected_history.append(item)
            history_tokens += item_tokens

    for item in reversed(selected_history):
        if item["role"] in {"user", "assistant"}:
            content = str(item["content"])
            if anonymize:
                content = anonymize_for_cloud(content)
            messages.append({"role": str(item["role"]), "content": content})

    # Context block appended to the user message — closer to model output improves recall
    # especially for small (4B) models where attention decays over long distances
    context_parts: list[str] = []
    if knowledge:
        # Base de conhecimento geral: não passa por anonimização (conteúdo de
        # referência, sem dados pessoais — o anonimizador mutilaria os exemplos)
        know_blocks = [
            f"({str(k.get('source', 'ref')).rsplit('.', 1)[0]})\n{str(k['content']).strip()}"
            for k in knowledge if str(k.get("content", "")).strip()
        ]
        if know_blocks:
            # Citação de fonte só em matéria oficial (lei, decreto, parecer...);
            # em conversa comum ela polui e vaza os bastidores da resposta
            if reasoning.is_official_matter(raw_prompt or user_prompt):
                instr = (
                    "[Referência oficial — use o que ajudar a responder e cite a "
                    "fonte entre parênteses ao usar, ex.: (direito_administrativo_pmc). "
                    "Se as referências não cobrirem o assunto, diga isso em vez de inventar. "
                    "Em caso de conflito, a referência prevalece sobre seu conhecimento prévio]\n"
                )
            else:
                instr = (
                    "[Apoio — use o que ajudar a responder, com naturalidade, sem citar "
                    "fontes nem mencionar que recebeu referências ou contexto]\n"
                )
            context_parts.append(instr + "\n\n".join(know_blocks))
    if global_text:
        context_parts.append(f"[Conhecimento acumulado — fundo de conversas anteriores; use com discrição, sem anunciar deduções sobre o usuário]\n{global_text}")
    if conv_text:
        context_parts.append(f"[Lembretes de conversas — apoio silencioso; não comente que os recebeu]\n{conv_text}")

    full_user = user_prompt
    if context_parts:
        full_user = user_prompt + "\n\n---\n" + "\n\n".join(context_parts)

    # Chain-of-thought forçado: perguntas analíticas ganham um andaime de
    # raciocínio no fim do contexto, onde a atenção de modelos pequenos é maior.
    # Detecta sobre a pergunta crua, não sobre o prompt com trechos de arquivos.
    scaffold = reasoning.build_scaffold(raw_prompt or user_prompt)
    if scaffold:
        full_user += scaffold

    messages.append({"role": "user", "content": full_user})
    return messages

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template_string(HTML)


@app.route("/api/agent-files/<file_id>", methods=["GET", "POST"])
def download_agent_file(file_id: str) -> Response | tuple[Response, int] | tuple[str, int]:
    if flask_request.method == "POST":
        saved_path = database.save_agent_file_to_workspace(file_id)
        if saved_path:
            try:
                with database.get_db() as conn:
                    row = conn.execute(
                        "SELECT conversation_id FROM agent_files WHERE id = ?", (file_id,)
                    ).fetchone()
                if row:
                    database.sync_workspace(str(row["conversation_id"]))
            except ValueError:
                pass
            return jsonify({
                "ok": True,
                "saved_to_workspace": True,
                "path": str(saved_path),
                "message": f"Arquivo salvo no workspace: {saved_path.name}",
            })
        return jsonify({"error": "Nenhum workspace vinculado ou arquivo indisponível."}), 400

    with database.get_db() as conn:
        row = conn.execute("SELECT * FROM agent_files WHERE id = ?", (file_id,)).fetchone()
    if not row:
        return "Arquivo não encontrado.", 404
    folder_id = str(row["folder_id"] or "agent")
    stored = str(row["stored_name"])
    path = Path(stored) if Path(stored).is_absolute() else AGENT_FILES_DIR / folder_id / stored
    if not path.exists():
        return "Arquivo removido do servidor.", 404
    return send_from_directory(
        str(path.parent), path.name,
        download_name=str(row["filename"]),
        as_attachment=True,
        mimetype=str(row["mime"]) or "application/octet-stream",
    )


@app.route("/uploads/<attachment_id>")
def download_upload(attachment_id: str) -> Response | tuple[str, int]:
    with database.get_db() as conn:
        row = conn.execute(
            "SELECT stored_name, original_name FROM attachments WHERE id = ?", (attachment_id,)
        ).fetchone()
    if not row:
        return "Arquivo não encontrado.", 404
    return send_from_directory(str(UPLOAD_DIR), str(row["stored_name"]), download_name=str(row["original_name"]))


@app.route("/api/personalities")
def api_personalities() -> Response:
    return jsonify({"personalities": list_personalities()})


@app.route("/api/status")
def api_status() -> Response:
    stats = database.get_memory_stats()
    stats["loaded_models"] = ai_engine.get_loaded_models()
    stats["ollama_url"] = ai_engine.OLLAMA_URL
    return jsonify(stats)


@app.route("/api/global-insights")
def api_global_insights() -> Response:
    insights = database.get_all_global_insights()
    return jsonify({"insights": insights, "count": len(insights)})


@app.route("/api/conversations/<conv_id>/export")
def api_export_conversation(conv_id: str) -> Response | tuple[str, int]:
    conv = database.get_or_create_conversation(conv_id)
    messages = database.get_messages(conv_id)
    if not messages:
        return "Conversa vazia.", 404

    lines_md: list[str] = [
        f"# {conv['title']}",
        f"Data: {conv.get('created_at', '?')}  |  Personalidade: {conv.get('personality', 'default')}",
        "",
        "---",
        "",
        "## Mensagens",
        "",
    ]
    for msg in messages:
        role_label = "**Você**" if msg["role"] == "user" else "**Assistente**"
        ts = str(msg.get("created_at", "")).replace("T", " ")[:16]
        lines_md.append(f"{role_label} _{ts}_")
        lines_md.append("")
        lines_md.append(str(msg["content"]))
        if msg.get("attachments"):
            for att in msg["attachments"]:
                lines_md.append(f"> 📎 {att['name']}")
        lines_md.append("")
        lines_md.append("---")
        lines_md.append("")

    with database.get_db() as conn:
        mems = conn.execute(
            "SELECT content, source_role FROM memories WHERE conversation_id = ? "
            "ORDER BY importance DESC, created_at DESC LIMIT 30",
            (conv_id,),
        ).fetchall()
        insights_rows = conn.execute(
            "SELECT content FROM insights WHERE conversation_id = ? ORDER BY created_at DESC",
            (conv_id,),
        ).fetchall()

    if mems:
        lines_md += ["## Memórias desta conversa", ""]
        for m in mems:
            lines_md.append(f"- `{m['source_role']}` {m['content'][:200]}")
        lines_md.append("")

    if insights_rows:
        lines_md += ["## Insights gerados", ""]
        for ins in insights_rows:
            lines_md.append(f"- {ins['content']}")
        lines_md.append("")

    filename = f"marcellus-conversa-{conv_id[:8]}.md"
    return Response(
        "\n".join(lines_md),
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/models")
def api_models() -> Response:
    models = ai_engine.list_local_models()
    warning = "" if models else (
        f"Não foi possível conectar ao Ollama em {ai_engine.OLLAMA_URL}. "
        "Para modelos locais, inicie com: ollama serve"
    )
    default = ai_engine.DEFAULT_MODEL or (models[0] if models else "")
    return jsonify({"models": models, "default_model": default,
                    "ollama_url": ai_engine.OLLAMA_URL, "warning": warning})


@app.route("/api/cloud-models", methods=["POST"])
def api_cloud_models() -> Response | tuple[Response, int]:
    body = flask_request.get_json(silent=True) or {}
    api_key = str(body.get("api_key", "")).strip()
    try:
        models = ai_engine.list_cloud_models(api_key)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"models": models})


@app.route("/api/select-folder", methods=["POST"])
def api_select_folder() -> Response | tuple[Response, int]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Selecionar pasta do workspace")
        root.destroy()
    except Exception as exc:
        return jsonify({
            "error": (
                "Não consegui abrir o seletor de pasta neste ambiente. "
                f"Erro: {exc}"
            )
        }), 500
    return jsonify({"path": path or ""})


@app.route("/api/folders", methods=["GET", "POST"])
def api_folders() -> Response:
    if flask_request.method == "POST":
        body = flask_request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip() or "Nova pasta"
        folder = database.create_folder(name)
        return jsonify({"folder": folder, "folders": database.get_folders()})
    return jsonify({"folders": database.get_folders()})


@app.route("/api/folders/<folder_id>", methods=["PATCH", "DELETE"])
def api_folder(folder_id: str) -> Response | tuple[Response, int]:
    if flask_request.method == "PATCH":
        body = flask_request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return jsonify({"error": "Nome vazio."}), 400
        database.rename_folder(folder_id, name)
        return jsonify({"ok": True})
    database.delete_folder(folder_id)
    return jsonify({"ok": True})


@app.route("/api/conversations", methods=["GET", "POST"])
def api_conversations() -> Response:
    if flask_request.method == "POST":
        body = flask_request.get_json(silent=True) or {}
        personality = str(body.get("personality", "atlas"))
        conv = database.get_or_create_conversation(title="Nova conversa", personality=personality)
        return jsonify({"conversation": conv, "conversations": database.get_conversations()})
    return jsonify({"conversations": database.get_conversations()})


@app.route("/api/conversations/<conv_id>", methods=["GET", "PATCH", "DELETE"])
def api_conversation(conv_id: str) -> Response | tuple[Response, int]:
    conv = database.get_or_create_conversation(conv_id)

    if flask_request.method == "PATCH":
        body = flask_request.get_json(silent=True) or {}
        updates: dict[str, Any] = {}
        if "title" in body:
            title = str(body["title"]).strip()
            if not title:
                return jsonify({"error": "Título vazio."}), 400
            updates["title"] = title
        if "personality" in body:
            updates["personality"] = str(body["personality"])
        if "folder_id" in body:
            updates["folder_id"] = body["folder_id"]  # None removes from folder
        database.update_conversation(conv_id, **updates)
        conv.update(updates)
        return jsonify({"conversation": conv})

    if flask_request.method == "DELETE":
        database.delete_conversation(conv_id)
        return jsonify({"ok": True})

    return jsonify({
        "conversation": conv,
        "messages": database.get_messages(conv_id),
        "workspace": database.workspace_summary(conv_id),
    })


@app.route("/api/search")
def api_search() -> Response:
    q = flask_request.args.get("q", "").strip()
    return jsonify({"results": database.search_all_conversations(q)})


@app.route("/api/messages/<message_id>/feedback", methods=["POST"])
def api_message_feedback(message_id: str) -> Response | tuple[Response, int]:
    data = flask_request.get_json(silent=True) or {}
    value = str(data.get("value", ""))
    if value not in ("", "up", "down"):
        return jsonify({"error": "value deve ser '', 'up' ou 'down'."}), 400
    if not database.set_message_feedback(message_id, value):
        return jsonify({"error": "Mensagem não encontrada."}), 404
    return jsonify({"ok": True, "feedback": value})


@app.route("/api/conversations/<conv_id>/workspace", methods=["PUT"])
def api_workspace(conv_id: str) -> Response | tuple[Response, int]:
    database.get_or_create_conversation(conv_id)
    body = flask_request.get_json(silent=True) or {}
    path = str(body.get("path", "")).strip()
    try:
        summary = database.set_workspace_path(conv_id, path)
        if path:
            summary = database.sync_workspace(conv_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "workspace": summary,
        "conversation": database.get_or_create_conversation(conv_id),
        "conversations": database.get_conversations(),
    })


@app.route("/api/conversations/<conv_id>/workspace/sync", methods=["POST"])
def api_workspace_sync(conv_id: str) -> Response | tuple[Response, int]:
    database.get_or_create_conversation(conv_id)
    try:
        summary = database.sync_workspace(conv_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "workspace": summary,
        "conversation": database.get_or_create_conversation(conv_id),
        "conversations": database.get_conversations(),
    })


@app.route("/api/conversations/<conv_id>/chat", methods=["POST"])
def api_chat(conv_id: str) -> Response | tuple[Response, int]:
    from werkzeug.datastructures import FileStorage

    conv = database.get_or_create_conversation(conv_id)
    prompt = flask_request.form.get("prompt", "").strip()
    model = flask_request.form.get("model", "").strip()
    provider = flask_request.form.get("provider", "local").strip()
    api_key = flask_request.form.get("api_key", "").strip()
    privacy_mode = flask_request.form.get("privacy_mode", "0") == "1"
    rigor_mode = flask_request.form.get("rigor_mode", "0") == "1"
    web_mode = flask_request.form.get("web_mode", "0") == "1"
    personality_id = flask_request.form.get("personality", conv.get("personality", "atlas"))
    files = cast(list[FileStorage], flask_request.files.getlist("files"))

    if not prompt and not files:
        return jsonify({"error": "Envie uma mensagem ou pelo menos um arquivo."}), 400
    if not model:
        return jsonify({"error": "Selecione um modelo."}), 400

    user_mid = database.add_message(conv["id"], "user", prompt or "[arquivos anexados]")
    attachments = database.store_attachments(user_mid, files, conv["id"])
    workspace_metrics: dict[str, Any] = {}
    if conv.get("workspace_path"):
        try:
            workspace_metrics = database.sync_workspace(conv["id"])
        except ValueError:
            workspace_metrics = {"workspace_error": "A pasta vinculada não existe mais."}
    full_prompt, attachment_metrics = build_user_prompt(prompt, attachments, provider, conv["id"])

    # Fase 6: pergunta de cálculo → resultado exato executado em Python,
    # injetado como fato — o modelo apresenta, não chuta
    calc_note = math_tool.solve(prompt)
    if calc_note:
        full_prompt += (
            f"\n\n[Cálculo verificado por execução — use exatamente este resultado: {calc_note}]"
        )

    # Fase 7a: toggle 🌐 — código busca na web (LexML + geral), modelo só lê
    if web_mode:
        web_ctx = web_search.build_web_context(prompt)
        if web_ctx:
            full_prompt += (
                "\n\n[Referências da web — podem estar desatualizadas ou imprecisas; "
                "cite o domínio ou link ao usar; se contradisserem a base local, "
                "aponte a divergência]\n" + web_ctx
            )

    if conv["title"] == "Nova conversa":
        title = summarize_title(prompt)
        database.update_conversation(conv["id"], title=title)
        conv["title"] = title

    if personality_id != conv.get("personality"):
        database.update_conversation(conv["id"], personality=personality_id)
        conv["personality"] = personality_id

    is_cloud = provider == "cloud-direct"
    privacy_applied = is_cloud and privacy_mode
    memories = mem.retrieve(
        full_prompt,
        limit=CLOUD_RETRIEVAL_LIMIT if is_cloud else LOCAL_RETRIEVAL_LIMIT,
        char_budget=CLOUD_RETRIEVAL_CHAR_BUDGET if is_cloud else LOCAL_RETRIEVAL_CHAR_BUDGET,
    )
    database.sync_knowledge()  # throttled: só reindexa se knowledge/ mudou
    kn_limit = CLOUD_KNOWLEDGE_LIMIT if is_cloud else LOCAL_KNOWLEDGE_LIMIT
    kn_budget = CLOUD_KNOWLEDGE_CHAR_BUDGET if is_cloud else LOCAL_KNOWLEDGE_CHAR_BUDGET
    knowledge = database.search_knowledge(
        tokenize(prompt), limit=kn_limit, char_budget=kn_budget, query_text=prompt,
    )
    if not knowledge:
        # Resgate de recall: pergunta coloquial sem match direto → o modelo
        # pequeno gera termos formais/sinônimos e a busca roda de novo.
        # Só paga a latência (~segundos) quando a busca direta falhou.
        extra_terms = query_expand.expand_query(prompt)
        if extra_terms:
            knowledge = database.search_knowledge(
                tokenize(prompt) + extra_terms,
                limit=kn_limit, char_budget=kn_budget,
                query_text=prompt + "\n" + " ".join(extra_terms),
            )
    ollama_messages = build_ollama_messages(
        conv["id"], full_prompt, memories,
        personality_id=personality_id, anonymize=privacy_applied, provider=provider,
        raw_prompt=prompt, knowledge=knowledge,
    )
    prompt_tokens_estimate = estimate_message_tokens(ollama_messages)
    context_limit = CONTEXT_LIMIT_CLOUD if is_cloud else CONTEXT_LIMIT_LOCAL
    context_pct = min(100, round(prompt_tokens_estimate / context_limit * 100))
    reasoning_profile = ai_engine.get_reasoning_profile(model, provider)

    def line(ev: dict[str, Any]) -> str:
        return json.dumps(ev, ensure_ascii=False) + "\n"

    def generate_events() -> Generator[str]:
        answer_parts: list[str] = []
        started_at = time.perf_counter()
        first_token_at: float | None = None
        yield line({"type": "meta", "conversation": conv, "conversations": database.get_conversations()})

        try:
            gen_options = reasoning.task_options(prompt) if provider == "local" else None
            for chunk in ai_engine.stream_chat(model, ollama_messages, provider, api_key, options=gen_options):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                answer_parts.append(chunk)
                yield line({"type": "chunk", "content": chunk})

            answer = "".join(answer_parts).strip()

            # Auto-continue if response was cut off mid-sentence (one attempt only)
            if ai_engine.is_response_truncated(answer):
                cont_messages = ollama_messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": "Continue exatamente de onde parou, sem repetir."},
                ]
                cont_parts: list[str] = []
                for chunk in ai_engine.stream_chat(model, cont_messages, provider, api_key, options=gen_options):
                    cont_parts.append(chunk)
                    yield line({"type": "chunk", "content": chunk})
                if cont_parts:
                    answer = (answer + " " + "".join(cont_parts)).strip()

            # --- Modo rigoroso: verifica o rascunho contra as referências ---
            # Segundo passe pelo mesmo modelo, temperatura baixa. Vale a
            # latência dobrada quando a resposta fundamenta parecer/ato.
            verification_applied = False
            if rigor_mode and knowledge and provider == "local" and answer.strip():
                yield line({"type": "status",
                            "message": "Modo rigoroso: verificando a resposta contra as referências..."})
                refs_txt = "\n\n".join(
                    f"({str(k.get('source', 'ref')).rsplit('.', 1)[0]})\n{str(k['content']).strip()}"
                    for k in knowledge if str(k.get("content", "")).strip()
                )[:6000]
                verify_prompt = (
                    "Você é um revisor técnico. Compare o RASCUNHO com as REFERÊNCIAS.\n"
                    "Corrija apenas afirmações que contradigam as referências e remova "
                    "citações de artigos ou números de norma que não constem nelas, "
                    "a menos que sejam de conhecimento certo. Mantenha o estilo e todo "
                    "o restante do texto. Se o rascunho já estiver correto, devolva-o "
                    "igual. Responda SOMENTE com a versão final, sem comentários.\n\n"
                    f"PERGUNTA:\n{prompt[:800]}\n\n"
                    f"REFERÊNCIAS:\n{refs_txt}\n\n"
                    f"RASCUNHO:\n{answer[:6000]}"
                )
                verified = ai_engine.generate_text(
                    model, verify_prompt, timeout=240,
                    options={"temperature": 0.2, "num_ctx": ai_engine.OLLAMA_NUM_CTX},
                )
                # Proteção: verificação vazia ou encolhida demais = falhou; mantém rascunho
                if len(verified) >= 0.4 * len(answer):
                    answer = verified
                    verification_applied = True

            # --- Parse and generate any files the AI embedded in its response ---
            cleaned_answer, file_blocks = file_gen.parse_file_blocks(answer)
            generated_files: list[dict] = []

            if file_blocks:
                answer = cleaned_answer
                ws_folder_id = str(conv.get("folder_id") or "agent")
                ws_dir = AGENT_FILES_DIR / ws_folder_id

                # Store message first so we can link files to it
                asst_mid = database.add_message(conv["id"], "assistant", answer)

                file_pairs: list[tuple[str, Any]] = []
                for block in file_blocks:
                    result = file_gen.generate_file(block["filename"], block["content"], ws_dir)
                    if result["ok"]:
                        file_path = ws_dir / result["stored_name"]
                        fid = database.store_agent_file(
                            conv["id"], asst_mid, conv.get("folder_id"),
                            block["filename"], result["stored_name"],
                            result["mime"], result["size"],
                        )
                        result["id"] = fid
                        file_pairs.append((block["filename"], file_path))
                    generated_files.append(result)

                # Auto-bundle ZIP when 2+ real files were created
                if len(file_pairs) >= 2:
                    zip_name = f"arquivos_{now_iso()[:10]}.zip"
                    zip_path = ws_dir / zip_name
                    file_gen.bundle_zip(file_pairs, zip_path)
                    if zip_path.exists():
                        zid = database.store_agent_file(
                            conv["id"], asst_mid, conv.get("folder_id"),
                            zip_name, zip_name,
                            "application/zip", zip_path.stat().st_size,
                        )
                        generated_files.append({
                            "filename": zip_name, "id": zid,
                            "mime": "application/zip", "ok": True,
                        })

            else:
                asst_mid = database.add_message(conv["id"], "assistant", answer)

            mem.remember(conv["id"], user_mid, "user", full_prompt)
            mem.remember(conv["id"], asst_mid, "assistant", answer)

            stats = database.get_memory_stats()
            finished_at = time.perf_counter()
            total_seconds = round(finished_at - started_at, 2)
            first_token_seconds = (
                round(first_token_at - started_at, 2) if first_token_at is not None else None
            )
            yield line({
                "type": "done",
                "conversation": conv,
                "conversations": database.get_conversations(),
                "messages": database.get_messages(conv["id"]),
                "memories_used": len(memories),
                "memory_count": stats["memory_count"],
                "insight_count": stats["insight_count"],
                "privacy_applied": privacy_applied,
                "metrics": {
                    "prompt_tokens_estimate": prompt_tokens_estimate,
                    "context_pct": context_pct,
                    "response_seconds": total_seconds,
                    "first_token_seconds": first_token_seconds,
                    "reasoning": reasoning_profile,
                    "rigor": verification_applied,
                    "workspace_added": int(workspace_metrics.get("added", 0) or 0),
                    "workspace_updated": int(workspace_metrics.get("updated", 0) or 0),
                    "workspace_removed": int(workspace_metrics.get("removed", 0) or 0),
                    **attachment_metrics,
                },
            })

            # Background consolidation — runs after response is delivered
            threading.Thread(
                target=mem.maybe_consolidate,
                args=(conv["id"], model),
                daemon=True,
            ).start()

        except Exception as exc:
            yield line({"type": "error", "error": f"Falha ao chamar o Ollama: {exc}"})

    return Response(
        stream_with_context(generate_events()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

@app.route("/api/execute-code", methods=["POST"])
def api_execute_code() -> Response | tuple[Response, int]:
    import code_exec
    body = flask_request.get_json(silent=True) or {}
    code = str(body.get("code", "")).strip()
    if not code:
        return jsonify({"error": "Código vazio."}), 400
    result = code_exec.execute_python(code)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

database.init_db()

# Indexa a base de conhecimento geral e os embeddings do few-shot em
# background (embeddings via Ollama podem demorar na primeira vez;
# não pode travar o startup)
threading.Thread(
    target=lambda: (
        database.sync_knowledge(force=True),
        few_shot.warm_cache(),
        query_expand.warm(),
        mem.backfill_embeddings(),
    ),
    daemon=True,
).start()

if __name__ == "__main__":
    import os as _os
    port = int(_os.environ.get("PORT", 5000))
    print(f"Marcellus — Interface: http://127.0.0.1:{port}")
    print(f"Ollama esperado em: {ai_engine.OLLAMA_URL}")
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)
