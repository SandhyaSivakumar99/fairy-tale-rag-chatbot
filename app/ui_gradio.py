import gradio as gr
from app.rag_pipeline import qa_chain

# 🤖 Response Logic
def respond_to_user(message, history):
    try:
        message_lower = message.lower().strip()
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "what's up", "how are you"]
        exit_phrases = ["bye", "goodbye", "see you later", "exit"]

        if message_lower in greetings:
            return "🧚‍♀️ Hello! Ask me anything about fairy tales and I’ll do my best to help!"
        if message_lower in exit_phrases:
            return "👋 Bye! Have a magical day! 🌟"

        response = qa_chain.invoke({"question": message})
        answer = response.get("answer", "") or response.get("result", "")

        if "don't know" in answer.lower() or "not sure" in answer.lower():
            answer += " 😊 I'm sorry, I don't know the answer to this question. But I'm always learning!"

        return answer + "\n🧙‍♀️ Thanks for asking!\nDo you want to ask anything else?"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ Error:\n{str(e)}"

# 🧚 Launch Gradio UI
def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🧚‍♀️ Welcome to Your Magical Fairy Tale Chatbot!")
        gr.Markdown("Talk to classic fairy tales like never before ✨ Ask about plots, characters, morals, and more.")
        gr.Image("screenshots/welcome_ui.png", height=278, width=500)
        gr.ChatInterface(
            fn=respond_to_user,
            title="🧚 Fairy Tale RAG Chatbot",
            description="Ask anything about your favourite fairy tales!",
            examples=["Does the little mermaid sing?", "Who helped Rapunzel escape?"],
            type="messages"
        )
    demo.launch(share=True, inline=False)

# 🚀 Run this file directly to launch
if __name__ == "__main__":
    launch_ui()
