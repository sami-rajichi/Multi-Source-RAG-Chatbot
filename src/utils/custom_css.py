CSS_STYLE = """
    <style>
        /* --- General --- */
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; }
        .main > div:first-child { max-width: 900px; margin: auto; padding: 1.5rem 1rem 5rem 1rem; } /* Increased bottom padding for chat input */
        .stApp > header { background-color: rgba(255,255,255,0.8); backdrop-filter: blur(5px); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        hr { border-top: 1px solid #e9ecef; margin: 1rem 0; }

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) { margin-left: auto; }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) > div[data-testid="stChatMessageContent"]{ background-color: #007bff; color: white; }
        [data-testid="chatAvatarIcon-user"] { background-color: #0056b3 !important; color: white !important;}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) { margin-right: auto; }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) > div[data-testid="stChatMessageContent"]{ background-color: #f1f3f5; color: #343a40; }
        [data-testid="chatAvatarIcon-assistant"] { background-color: #495057 !important; color: white !important;}
        .stChatMessage .stCaption { font-size: 0.75rem; color: #6c757d; padding-top: 8px; text-align: right; }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) .stCaption { color: #b3d7ff; }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; padding-top: 1rem;}
        [data-testid="stSidebar"] .stTabs [data-testid="stMarkdownContainer"] p { font-weight: 600; font-size: 0.95rem; }
        [data-testid="stSidebar"] .stExpander { border: 1px solid #dee2e6; margin-bottom: 1rem; background-color: #fff; border-radius: 8px;}
        [data-testid="stSidebar"] .stExpander > summary { font-weight: 600; padding: 0.7rem 1rem !important; font-size: 0.95rem; }
        [data-testid="stSidebar"] .stButton>button { margin-top: 0.5rem;}

        /* --- Buttons --- */
        .stButton>button { border-radius: 8px; padding: 0.6rem 1rem; font-weight: 500; width: 100%; border: 1px solid #ced4da; transition: all 0.2s ease-in-out; }
        .stButton>button:hover { border-color: #adb5bd; background-color: #f1f3f5; }
        .stButton>button:active { transform: scale(0.98); }
        .stButton>button[kind="primary"] { background-color: #007bff; color: white; border-color: #007bff;}
        .stButton>button[kind="primary"]:hover { background-color: #0056b3; border-color: #0056b3;}
        .stButton>button:has(span>span:contains("Delete")) { background-color: #fdf2f2 !important; color: #dc3545 !important; border-color: #dc3545 !important;}
        .stButton>button:has(span>span:contains("Delete")):hover { background-color: #f8d7da !important; border-color: #b02a37 !important;}
        .stButton>button:has(span>span:contains("Rebuild")) { background-color: #f0f7ff !important; color: #17a2b8 !important; border-color: #17a2b8 !important;}
        .stButton>button:has(span>span:contains("Rebuild")):hover { background-color: #d1ecf1 !important; border-color: #117a8b !important;}

        /* --- Fixed Chat Input Container --- */
        .fixed-chat-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #e9ecef;
            z-index: 100;
        }
        .chat-input-wrapper {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .chat-input-wrapper .stTextInput {
            flex-grow: 1;
        }
        .chat-input-wrapper .stButton {
            margin: 0;
        }
        .chat-input-wrapper button {
            height: 44px !important;
            min-width: 44px !important;
            padding: 0 !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Chat history container with scroll */
        .chat-history-container {
            max-height: calc(100vh - 250px);
            overflow-y: auto;
            padding-bottom: 1rem;
        }
        
        /* Env var status styling */
        .env-var-status {
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }
        .env-var-status.valid {
            background-color: #e6f7ee;
            color: #0d6832;
        }
        .env-var-status.invalid {
            background-color: #fdf2f2;
            color: #9e2d2d;
        }
        .env-var-status.warning {
            background-color: #fff8e6;
            color: #8a6d3b;
        }
    </style>
    """