import os
import re


def insert_runllm_widget(html_content):
    # RunLLM Widget script to be inserted
    widget_script = """
    <!-- RunLLM Widget Script -->
    <script type="module" id="runllm-widget-script" src="https://widget.runllm.com" crossorigin="true" version="stable" runllm-keyboard-shortcut="Mod+j" runllm-name="SGLang Chatbot" runllm-position="BOTTOM_RIGHT" runllm-assistant-id="629" async></script>
    """

    # Find the closing body tag and insert the widget script before it
    return re.sub(r"</body>", f"{widget_script}\n</body>", html_content)


def main():
    # Get the build directory path
    build_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_build", "html"
    )
    index_file = os.path.join(build_dir, "index.html")

    # Process only index.html
    if os.path.exists(index_file):
        # Read the HTML file
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Insert the RunLLM widget
        modified_content = insert_runllm_widget(content)

        # Write back the modified content
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(modified_content)
    else:
        print(f"Index file not found: {index_file}")


if __name__ == "__main__":
    main()
