#!/usr/bin/env python3
"""
VoLens Public Server Launcher
Creates a public URL accessible from anywhere (phone, home computer, etc.)
"""

import subprocess
import threading
import time
import sys
import os

def start_dashboard():
    """Start the dashboard server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "dashboard.py"])

def start_tunnel():
    """Start ngrok tunnel and display public URL"""
    from pyngrok import ngrok, conf
    
    # Wait for dashboard to start
    time.sleep(3)
    
    print("\n" + "="*60)
    print("🌍 Creating Public Tunnel...")
    print("="*60)
    
    try:
        # Create tunnel to port 8050
        public_url = ngrok.connect(8050)
        
        print("\n" + "🎉 " + "="*56 + " 🎉")
        print("   PUBLIC ACCESS ENABLED!")
        print("="*60)
        print(f"\n📱 Access from ANYWHERE at:")
        print(f"\n   {public_url}")
        print(f"\n   👆 Open this URL on your iPhone or home computer!")
        print("\n" + "="*60)
        print("\n💡 Tips:")
        print("   • This URL works from any device with internet")
        print("   • Share it via text/email to your phone")
        print("   • URL changes each time you restart")
        print("   • Press Ctrl+C to stop the server")
        print("\n" + "="*60 + "\n")
        
        # Keep the tunnel alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down tunnel...")
            ngrok.kill()
            
    except Exception as e:
        print(f"\n⚠️  Error creating tunnel: {e}")
        print("\n📝 You may need to sign up for a free ngrok account:")
        print("   1. Go to https://ngrok.com/signup")
        print("   2. Sign up for free")
        print("   3. Copy your authtoken from the dashboard")
        print("   4. Run: ngrok config add-authtoken YOUR_TOKEN")
        print("\n   Then try again!")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 VoLens - Stock Volatility Predictor")
    print("   PUBLIC ACCESS MODE")
    print("="*60)
    
    # Start dashboard in a separate thread
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Start the tunnel
    start_tunnel()

