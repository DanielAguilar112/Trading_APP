"""
chatbot.py - AI Trading Assistant (Free Local Version using Ollama/Llama3)
Run: python chatbot.py
Then open http://localhost:5000
"""
import os, json, threading
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import ollama

from config import WATCHLIST, CAPITAL, CONFIDENCE_THRESHOLD, OUTPUT_DIR, MODEL_DIR
import config as cfg

app = Flask(__name__)
app.config["SECRET_KEY"] = "trading_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
conversation_history = []

SYSTEM_PROMPT = """You are an AI trading assistant with real-time access to a live algorithmic trading system. Be helpful, direct and concise. You help with signals, portfolio performance, model accuracy, settings, and trading education. Never guarantee profits. System data is injected automatically with each message."""

def get_signals_data():
    try:
        f = OUTPUT_DIR / "signals_latest.json"
        return json.loads(f.read_text()) if f.exists() else {"status": "No signals yet"}
    except: return {"status": "Error reading signals"}

def get_portfolio_data():
    try:
        f = OUTPUT_DIR / "dry_run_state.json"
        if f.exists(): return json.loads(f.read_text())
        f2 = OUTPUT_DIR / "trade_journal.json"
        if f2.exists():
            t = json.loads(f2.read_text())
            return {"trade_count": len(t), "recent": t[-3:]}
    except: pass
    return {"status": "No portfolio data yet"}

def get_journal_data():
    try:
        f = OUTPUT_DIR / "trade_journal.json"
        if f.exists():
            t = json.loads(f.read_text())
            return t[-20:] if t else []
    except: pass
    return []

def get_model_data():
    try:
        f = MODEL_DIR / "registry.json"
        if f.exists():
            d = json.loads(f.read_text())
            return {k: {"trained": v.get("last_trained","")[:10],
                        "acc": {m: round(mv.get("accuracy",0)*100,1) for m,mv in v.get("metrics",{}).items()}}
                    for k,v in list(d.items())[:6]}
    except: pass
    return {}

def get_context():
    signals = get_signals_data()
    sig_list = [{"ticker":s.get("ticker"),"action":s.get("action"),"conf":s.get("confidence_pct"),"p_up":s.get("p_up")}
                for s in signals.get("signals",[])] if "signals" in signals else []
    journal = get_journal_data()
    return f"""SYSTEM ({datetime.now().strftime('%Y-%m-%d %H:%M')}):
Config: {len(WATCHLIST)} tickers, threshold={CONFIDENCE_THRESHOLD*100:.0f}%, capital=${CAPITAL:,.0f}, max_risk={cfg.MAX_RISK_PER_TRADE*100:.0f}%
Signals: {json.dumps(sig_list)}
Portfolio: {json.dumps(get_portfolio_data())}
Recent trades ({len(journal)} total): {json.dumps(journal[-5:])}
Models: {json.dumps(get_model_data())}"""

def chat(user_msg):
    global conversation_history
    conversation_history.append({"role":"user","content": user_msg + "\n\n[DATA]\n" + get_context()})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    try:
        resp = ollama.chat(model="llama3",
                           messages=[{"role":"system","content":SYSTEM_PROMPT}] + conversation_history)
        reply = resp["message"]["content"]
        conversation_history[-1]["content"] = user_msg
        conversation_history.append({"role":"assistant","content":reply})
        return reply
    except Exception as e:
        err = str(e)
        if "connect" in err.lower() or "refused" in err.lower():
            return "Ollama is not running. Open the Ollama app from your Windows start menu, then try again."
        return f"Error: {err}"

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/signals")
def api_signals(): return jsonify(get_signals_data())

@app.route("/api/portfolio")
def api_portfolio(): return jsonify(get_portfolio_data())

@app.route("/api/journal")
def api_journal(): return jsonify(get_journal_data())

@app.route("/api/config")
def api_config():
    return jsonify({"watchlist":WATCHLIST,"confidence_threshold":CONFIDENCE_THRESHOLD,
                    "capital":CAPITAL,"max_risk_per_trade":cfg.MAX_RISK_PER_TRADE,
                    "max_portfolio_heat":cfg.MAX_PORTFOLIO_HEAT})

@socketio.on("send_message")
def handle_message(data):
    msg = data.get("message","").strip()
    if not msg: return
    emit("user_message", {"message":msg,"time":datetime.now().strftime("%H:%M")})
    emit("typing", {"status":True})
    def respond():
        reply = chat(msg)
        socketio.emit("bot_message", {"message":reply,"time":datetime.now().strftime("%H:%M")})
        socketio.emit("typing", {"status":False})
    threading.Thread(target=respond, daemon=True).start()

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Trading Assistant</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
.header{background:#1a1a1a;border-bottom:1px solid #2a2a2a;padding:14px 20px;display:flex;align-items:center;justify-content:space-between}
.logo{width:36px;height:36px;background:linear-gradient(135deg,#00d4aa,#0088ff);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-right:12px}
.htitle{font-size:16px;font-weight:600;color:#fff}
.hsub{font-size:12px;color:#666;margin-top:1px}
.badge{background:#0d2818;border:1px solid #1a5c33;color:#00d4aa;font-size:11px;padding:4px 10px;border-radius:20px;display:flex;align-items:center;gap:5px}
.dot{width:6px;height:6px;border-radius:50%;background:#00d4aa;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.stats{background:#141414;border-bottom:1px solid #222;padding:10px 20px;display:flex;gap:24px;overflow-x:auto}
.stat{display:flex;flex-direction:column;gap:2px;min-width:fit-content}
.slabel{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:.05em}
.sval{font-size:14px;font-weight:600;color:#fff}
.sval.g{color:#00d4aa}.sval.r{color:#ff4444}.sval.y{color:#ffaa00}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:260px;background:#141414;border-right:1px solid #222;overflow-y:auto;display:flex;flex-direction:column}
.ssec{padding:14px 16px;border-bottom:1px solid #1e1e1e}
.stitle{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.sig{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #1a1a1a}
.sig:last-child{border-bottom:none}
.sticker{font-size:13px;font-weight:600;color:#fff}
.sinfo{display:flex;align-items:center;gap:8px}
.sconf{font-size:11px;color:#666}
.pill{font-size:10px;padding:2px 7px;border-radius:3px;font-weight:600}
.pill-buy{background:#0d2818;color:#00d4aa;border:1px solid #1a5c33}
.pill-sell{background:#2a0d0d;color:#ff4444;border:1px solid #5c1a1a}
.pill-hold{background:#1e1e1e;color:#888;border:1px solid #333}
.qbtns{display:flex;flex-direction:column;gap:6px}
.qbtn{background:#1e1e1e;border:1px solid #2a2a2a;color:#ccc;padding:8px 12px;border-radius:6px;font-size:12px;cursor:pointer;text-align:left;transition:all .15s}
.qbtn:hover{background:#252525;border-color:#00d4aa;color:#00d4aa}
.chat{flex:1;display:flex;flex-direction:column;overflow:hidden}
.msgs{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.msg{display:flex;gap:10px;max-width:85%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.msg.bot{align-self:flex-start}
.av{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.av.bot{background:linear-gradient(135deg,#00d4aa22,#0088ff22);border:1px solid #00d4aa44;color:#00d4aa}
.av.user{background:#252525;border:1px solid #333;color:#888}
.bub{padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.6}
.bub.bot{background:#1a1a1a;border:1px solid #252525;color:#e0e0e0;border-radius:4px 12px 12px 12px}
.bub.user{background:#0d2818;border:1px solid #1a5c33;color:#e0e0e0;border-radius:12px 4px 12px 12px}
.mtime{font-size:10px;color:#444;margin-top:4px}
.typing{display:flex;gap:4px;padding:12px 14px;background:#1a1a1a;border:1px solid #252525;border-radius:4px 12px 12px 12px;width:fit-content}
.tdot{width:6px;height:6px;border-radius:50%;background:#00d4aa;animation:t 1.2s infinite}
.tdot:nth-child(2){animation-delay:.2s}.tdot:nth-child(3){animation-delay:.4s}
@keyframes t{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
.iarea{padding:16px 20px;background:#141414;border-top:1px solid #222}
.iwrap{display:flex;gap:10px;align-items:flex-end}
.ibox{flex:1;background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;padding:10px 14px;color:#e0e0e0;font-size:14px;resize:none;outline:none;font-family:inherit;max-height:120px;line-height:1.5;transition:border-color .15s}
.ibox:focus{border-color:#00d4aa44}
.ibox::placeholder{color:#444}
.sbtn{width:40px;height:40px;background:linear-gradient(135deg,#00d4aa,#0088ff);border:none;border-radius:8px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:opacity .15s}
.sbtn:hover{opacity:.85}
.sbtn svg{width:16px;height:16px;fill:white}
.welcome{text-align:center;padding:40px 20px;color:#555}
.welcome h2{color:#888;font-size:18px;margin-bottom:8px}
.welcome p{font-size:13px;line-height:1.6;max-width:400px;margin:0 auto 12px}
.freebadge{background:#1a1a2e;border:1px solid #2a2a5c;color:#6688ff;font-size:11px;padding:4px 10px;border-radius:20px;display:inline-block}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:2px}
@media(max-width:768px){.sidebar{display:none}}
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center">
    <div class="logo">&#9650;</div>
    <div><div class="htitle">AI Trading Assistant</div><div class="hsub">Llama3 Local &bull; 100% Free</div></div>
  </div>
  <div class="badge"><div class="dot"></div>Live</div>
</div>
<div class="stats">
  <div class="stat"><div class="slabel">Portfolio</div><div class="sval" id="sPort">Loading...</div></div>
  <div class="stat"><div class="slabel">P&amp;L</div><div class="sval" id="sPnl">--</div></div>
  <div class="stat"><div class="slabel">Signals</div><div class="sval" id="sSig">--</div></div>
  <div class="stat"><div class="slabel">Trades</div><div class="sval" id="sTrades">--</div></div>
  <div class="stat"><div class="slabel">Threshold</div><div class="sval y" id="sConf">--</div></div>
  <div class="stat"><div class="slabel">Watchlist</div><div class="sval" id="sWatch">--</div></div>
</div>
<div class="main">
  <div class="sidebar">
    <div class="ssec">
      <div class="stitle">Live Signals</div>
      <div id="sigList"><div style="color:#555;font-size:12px">Loading...</div></div>
    </div>
    <div class="ssec">
      <div class="stitle">Quick Actions</div>
      <div class="qbtns">
        <button class="qbtn" onclick="q('What are the current signals?')">&#128200; Current signals</button>
        <button class="qbtn" onclick="q('How is my portfolio performing?')">&#128181; Portfolio P&amp;L</button>
        <button class="qbtn" onclick="q('Show me recent trades and how they did')">&#128203; Recent trades</button>
        <button class="qbtn" onclick="q('How accurate are the AI models right now?')">&#129302; Model accuracy</button>
        <button class="qbtn" onclick="q('Should I change settings to improve performance?')">&#9881; Optimize settings</button>
        <button class="qbtn" onclick="q('Why is the system not trading right now?')">&#10067; Why not trading?</button>
        <button class="qbtn" onclick="q('Which tickers look most promising right now?')">&#128269; Best opportunities</button>
      </div>
    </div>
  </div>
  <div class="chat">
    <div class="msgs" id="msgs">
      <div class="welcome">
        <div style="font-size:48px;margin-bottom:16px">&#129302;</div>
        <h2>AI Trading Assistant</h2>
        <p>Ask me anything about your trading system. I can see your live signals, portfolio, trades, and model performance in real time.</p>
        <div class="freebadge">Running locally on your machine &bull; 100% free</div>
      </div>
    </div>
    <div class="iarea">
      <div class="iwrap">
        <textarea class="ibox" id="inp" placeholder="Ask about signals, portfolio, trades..." rows="1"></textarea>
        <button class="sbtn" onclick="send()"><svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg></button>
      </div>
    </div>
  </div>
</div>
<script>
const socket=io();
const inp=document.getElementById('inp');
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,120)+'px'});
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
function send(){const m=inp.value.trim();if(!m)return;socket.emit('send_message',{message:m});inp.value='';inp.style.height='auto'}
function q(m){socket.emit('send_message',{message:m})}
socket.on('user_message',d=>addMsg('user',d.message,d.time));
socket.on('bot_message',d=>{rmTyping();addMsg('bot',d.message,d.time)});
socket.on('typing',d=>{if(d.status)showTyping();else rmTyping()});
function addMsg(role,text,time){
  const w=document.querySelector('.welcome');if(w)w.remove();
  const msgs=document.getElementById('msgs');
  const d=document.createElement('div');d.className='msg '+role;
  const fmt=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\n/g,'<br>').replace(/[*][*](.*?)[*][*]/g,'<strong>$1</strong>')
    .replace(/`(.*?)`/g,'<code style="background:#252525;padding:1px 5px;border-radius:3px;font-size:12px">$1</code>');
  d.innerHTML=`<div class="av ${role}">${role==='bot'?'&#129302;':'&#128100;'}</div><div><div class="bub ${role}">${fmt}</div><div class="mtime">${time}</div></div>`;
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function showTyping(){rmTyping();const msgs=document.getElementById('msgs');const d=document.createElement('div');d.className='msg bot';d.id='typ';d.innerHTML='<div class="av bot">&#129302;</div><div class="typing"><div class="tdot"></div><div class="tdot"></div><div class="tdot"></div></div>';msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight}
function rmTyping(){const e=document.getElementById('typ');if(e)e.remove()}
async function loadStats(){
  try{
    const[c,s,j]=await Promise.all([fetch('/api/config').then(r=>r.json()),fetch('/api/signals').then(r=>r.json()),fetch('/api/journal').then(r=>r.json())]);
    document.getElementById('sConf').textContent=(c.confidence_threshold*100).toFixed(0)+'%';
    document.getElementById('sWatch').textContent=c.watchlist.length+' tickers';
    document.getElementById('sTrades').textContent=j.length||0;
    if(s.signals){
      const buys=s.signals.filter(x=>x.action==='BUY').length;
      const sells=s.signals.filter(x=>x.action==='SELL').length;
      document.getElementById('sSig').textContent=buys+'B / '+sells+'S';
      const top=s.signals.filter(x=>x.action!=='HOLD').sort((a,b)=>b.confidence_pct-a.confidence_pct).slice(0,8);
      document.getElementById('sigList').innerHTML=top.length===0?'<div style="color:#555;font-size:12px">No actionable signals</div>':top.map(x=>`<div class="sig"><span class="sticker">${x.ticker}</span><div class="sinfo"><span class="sconf">${x.confidence_pct}%</span><span class="pill pill-${x.action.toLowerCase()}">${x.action}</span></div></div>`).join('');
    }
  }catch(e){}
  try{
    const p=await fetch('/api/portfolio').then(r=>r.json());
    if(p.account){
      const val=p.account.portfolio_value,pnl=p.account.unrealized_pnl;
      document.getElementById('sPort').textContent='$'+Number(val).toLocaleString();
      const el=document.getElementById('sPnl');
      el.textContent=(pnl>=0?'+':'')+' $'+Math.abs(Number(pnl)).toFixed(0);
      el.className='sval '+(pnl>=0?'g':'r');
    }else{document.getElementById('sPort').textContent='$100,000';document.getElementById('sPnl').textContent='Paper'}
  }catch(e){}
}
loadStats();setInterval(loadStats,30000);
</script>
</body>
</html>'''

if __name__ == "__main__":
    print("""
+----------------------------------------------+
|  AI TRADING CHATBOT  (Free - Llama3 Local)   |
|  Open http://localhost:5000 in browser        |
|  Make sure Ollama app is running first        |
|  Press Ctrl+C to stop                        |
+----------------------------------------------+
""")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)