"""
Smart Router: Automatically routes requests between local Ollama and remote SGLang.

Uses an LLM judge to classify tasks as simple or complex, then routes accordingly:
- Simple tasks → Local Ollama (fast response)
- Complex tasks → Remote SGLang (powerful model)

Usage:
    from sglang.srt.entrypoints.ollama.smart_router import SmartRouter

    router = SmartRouter(
        local_host="http://localhost:11434",
        remote_host="http://sglang-server:30001",
    )
    response = router.chat("Hello!")
"""

from typing import Optional

import ollama


class SmartRouter:
    """Routes requests between local Ollama and remote SGLang using LLM-based classification."""

    # Classification prompt for LLM judge
    CLASSIFICATION_PROMPT = """You are a task classifier. Classify the following user request into one of two categories.

Categories:
- SIMPLE: Quick responses, greetings, factual questions, definitions, translations, basic Q&A
- COMPLEX: Tasks requiring deep reasoning, multi-step analysis, long explanations, creative writing, detailed research

Reply with ONLY one word: either SIMPLE or COMPLEX.

User request: "{prompt}"

Category:"""

    def __init__(
        self,
        local_host: str = "http://localhost:11434",
        remote_host: str = "http://localhost:30001",
        local_model: str = "llama3.2",
        remote_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        judge_model: Optional[str] = None,
        judge_host: Optional[str] = None,
    ):
        """
        Initialize the smart router.

        Args:
            local_host: URL of local Ollama server
            remote_host: URL of remote SGLang server
            local_model: Model name for local Ollama
            remote_model: Model name for remote SGLang
            judge_model: Model for LLM-based classification (default: same as local_model)
            judge_host: Host for judge model (default: same as local_host)
        """
        self.local_client = ollama.Client(host=local_host)
        self.remote_client = ollama.Client(host=remote_host)
        self.local_model = local_model
        self.remote_model = remote_model

        # Judge model configuration
        self.judge_model = judge_model or local_model
        self.judge_host = judge_host or local_host
        self.judge_client = ollama.Client(host=self.judge_host)

    def _classify_with_llm(
        self, prompt: str, verbose: bool = False
    ) -> tuple[bool, str]:
        """
        Use LLM to classify the prompt.

        Returns:
            Tuple of (use_remote, reason)
        """
        try:
            classification_prompt = self.CLASSIFICATION_PROMPT.format(
                prompt=prompt[:500]  # Limit prompt length for classification
            )

            response = self.judge_client.chat(
                model=self.judge_model,
                messages=[{"role": "user", "content": classification_prompt}],
                options={"temperature": 0, "num_predict": 10},
            )

            result = response["message"]["content"].strip().upper()

            if verbose:
                print(f"[Router] LLM Judge: {result}")

            if "COMPLEX" in result:
                return True, "Complex task"
            else:
                return False, "Simple task"

        except Exception as e:
            if verbose:
                print(f"[Router] LLM Judge failed: {e}, defaulting to local")
            return False, "Judge failed, defaulting to local"

    def should_use_remote(self, prompt: str, verbose: bool = False) -> tuple[bool, str]:
        """
        Determine if the prompt should be routed to remote SGLang.

        Args:
            prompt: User's input prompt
            verbose: Print debug information

        Returns:
            Tuple of (should_use_remote, reason)
        """
        return self._classify_with_llm(prompt, verbose)

    def chat(
        self,
        prompt: str,
        messages: Optional[list] = None,
        verbose: bool = False,
        force_local: bool = False,
        force_remote: bool = False,
    ) -> dict:
        """
        Route the request and get response.

        Args:
            prompt: User's input (used if messages is None)
            messages: Full message history (overrides prompt if provided)
            verbose: Print routing decision
            force_local: Force use of local model
            force_remote: Force use of remote model

        Returns:
            Response dict with 'content', 'model', 'location', 'reason' keys
        """
        # Build messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
            check_prompt = prompt
        else:
            # Use the last user message for routing decision
            check_prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    check_prompt = msg.get("content", "")
                    break

        # Determine routing
        if force_remote:
            use_remote, reason = True, "Forced remote"
        elif force_local:
            use_remote, reason = False, "Forced local"
        else:
            use_remote, reason = self.should_use_remote(check_prompt, verbose)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] -> {location} | Model: {model}")

        try:
            response = client.chat(model=model, messages=messages)
            return {
                "content": response["message"]["content"],
                "model": model,
                "location": location,
                "reason": reason,
            }
        except Exception as e:
            # Fallback to the other option
            if verbose:
                print(f"[Router] {location} failed: {e}, falling back...")

            fallback_client = (
                self.remote_client if not use_remote else self.local_client
            )
            fallback_model = self.remote_model if not use_remote else self.local_model
            fallback_location = "Remote SGLang" if not use_remote else "Local Ollama"

            response = fallback_client.chat(model=fallback_model, messages=messages)
            return {
                "content": response["message"]["content"],
                "model": fallback_model,
                "location": fallback_location,
                "reason": f"Fallback from {location}",
            }

    def chat_stream(
        self,
        prompt: str,
        messages: Optional[list] = None,
        verbose: bool = False,
        force_local: bool = False,
        force_remote: bool = False,
    ):
        """
        Route the request and stream response.

        Yields:
            Response chunks
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
            check_prompt = prompt
        else:
            check_prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    check_prompt = msg.get("content", "")
                    break

        if force_remote:
            use_remote, reason = True, "Forced remote"
        elif force_local:
            use_remote, reason = False, "Forced local"
        else:
            use_remote, reason = self.should_use_remote(check_prompt, verbose)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] -> {location} | Model: {model}")

        for chunk in client.chat(model=model, messages=messages, stream=True):
            yield chunk


def main():
    """Interactive demo of the smart router."""
    print("=" * 60)
    print("Smart Router: Local Ollama <-> Remote SGLang")
    print("=" * 60)
    print("\nRouting strategy:")
    print("  LLM Judge classifies each request as SIMPLE or COMPLEX")
    print("  - SIMPLE tasks -> Local Ollama (fast)")
    print("  - COMPLEX tasks -> Remote SGLang (powerful)")
    print("\nType 'quit' to exit\n")

    router = SmartRouter(
        local_host="http://localhost:11434",
        remote_host="http://localhost:30001",
        local_model="llama3.2",
        remote_model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # Use streaming for real-time output
            print("\nAssistant: ", end="", flush=True)
            full_response = ""
            for chunk in router.chat_stream(
                prompt=user_input, messages=messages, verbose=True
            ):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            print("\n")

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
