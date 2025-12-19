import streamlit as st
import os
from client import MCPClient
from server import generate_campaign_job, TOOL_DEFINITION

# Configure Page
st.set_page_config(page_title="MCP Marketing Suite", page_icon="🚀", layout="wide")

# Initialize the MCP Client
@st.cache_resource
def get_client():
    return MCPClient(TOOL_DEFINITION, generate_campaign_job)

client = get_client()

# --- Sidebar UI ---
st.sidebar.title("🛠️ MCP Control Panel")
st.sidebar.info("Connected to: **Content_Gen_Service**")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Gemini 1.5 Flash")
st.sidebar.write("**Status:** Ready")

# --- Main UI ---
st.title("🎨 AI Multimodal Campaign Architect")
st.markdown("Enter your campaign goal below to generate copy, moodboards, and voiceover scripts.")

# Input Section
with st.container():
    campaign_goal = st.text_area(
        "What are we launching?", 
        placeholder="e.g., A luxury sustainable coffee brand for urban professionals.",
        help="Provide a high-level description of your brand or product."
    )
    
    generate_btn = st.button("🚀 Generate Full Campaign", type="primary")

# --- Execution Logic ---
if generate_btn:
    if not campaign_goal.strip():
        st.warning("Please enter a campaign goal first.")
    else:
        with st.status("🛠️ MCP Server working...", expanded=True) as status:
            st.write("Initializing request...")
            result_package = client.execute_tool(campaign_goal)
            
            if result_package["status"] == "success":
                res = result_package["result"]
                status.update(label="✅ Campaign Generated!", state="complete", expanded=False)
                
                st.divider()

                # --- UI Layout: Top Section (Headline & Pitch) ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header(f"📢 {res['headline']}")
                    st.subheader("Body Copy")
                    st.write(res['body_copy'])
                
                with col2:
                    st.subheader("🎯 Pitch Summary")
                    st.markdown(f"**Target:** {res['pitch']['target_audience']}")
                    st.markdown(f"**Core Message:** {res['pitch']['core_message']}")
                    st.caption(f"*Rationale:* {res['pitch']['design_rationale']}")

                st.divider()

                # --- UI Layout: Middle Section (Visuals) ---
                st.header("📸 Visual Concept & Moodboard")
                
                # Only show Generated Image if it exists
                if res.get('generated_image_local_path') and os.path.exists(res['generated_image_local_path']):
                    st.image(res['generated_image_local_path'], caption="AI Generated Hero Image", use_container_width=True)
                
                # Show Unsplash Moodboard
                if res.get('visual_moodboard_urls'):
                    cols = st.columns(len(res['visual_moodboard_urls']))
                    for idx, url in enumerate(res['visual_moodboard_urls']):
                        cols[idx].image(url, caption=f"Visual Ref {idx+1}")

                st.divider()

                # --- UI Layout: Bottom Section (Audio) ---
                st.header("🎙️ Audio Asset")
                a_col1, a_col2 = st.columns([1, 2])
                
                with a_col1:
                    if res.get('audio_file_path') and os.path.exists(res['audio_file_path']):
                        st.audio(res['audio_file_path'])
                    else:
                        st.error("Audio generation failed.")
                
                with a_col2:
                    st.info(f"**Voice Script:**\n\n {res['audio_script']}")

            else:
                status.update(label="❌ Error", state="error")
                st.error(f"Execution failed: {result_package['details']}")

# Footer
st.markdown("---")
st.caption("Powered by MCP Server Architecture • Gemini 1.5 Flash • Unsplash API")
