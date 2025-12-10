#!/usr/bin/env python3
"""
Setup Remote Access for VoLens
Run this once to configure ngrok for remote access
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║         🌍 VoLens Remote Access Setup                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  This will allow you to access VoLens from:                  ║
║    • Your iPhone                                             ║
║    • Your home computer                                      ║
║    • Anywhere with internet!                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📝 STEP 1: Get your free ngrok authtoken

   1. Go to: https://ngrok.com/signup
   2. Create a free account (Google/GitHub/email)
   3. After signup, go to: https://dashboard.ngrok.com/get-started/your-authtoken
   4. Copy your authtoken

""")

token = input("📋 Paste your ngrok authtoken here: ").strip()

if not token:
    print("\n❌ No token provided. Please try again.")
    exit(1)

try:
    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    print("\n✅ Token saved successfully!")
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🎉 Setup Complete!                                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To start VoLens with remote access, run:                    ║
║                                                              ║
║    python start_public.py                                    ║
║                                                              ║
║  You'll get a public URL like:                               ║
║    https://abc123.ngrok.io                                   ║
║                                                              ║
║  Open that URL on your iPhone or home computer!              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Please check your token and try again.")

