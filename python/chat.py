#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-CopyrightText: 2024 Gerhard Gappmeier <gappy1502@gmx.net>
#
# Unified chat client for Ollama and OpenAI APIs.
# Supports conversation context and multiline input.

import sys
import argparse
import httpx
import json
import asyncio
import datetime
import subprocess
import re
import shlex
from OllamaLogger import OllamaLogger
from OllamaCredentials import OllamaCredentials

# Try to import OpenAI SDK
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Default values
DEFAULT_PROVIDER = "ollama"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 600
# Default models if missing
DEFAULT_MODEL = "codellama:code"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
# default options if missing
DEFAULT_OPTIONS = '{ "temperature": 0, "top_p": 0.95 }'
# default parameters if options is given, but missing these entries
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 5000

log = None

# Command execution settings
DEFAULT_COMMAND_TIMEOUT = 30  # seconds
MAX_COMMAND_LENGTH = 1000  # characters
MAX_COMMAND_OUTPUT = 10000  # characters

def execute_command(command, timeout=DEFAULT_COMMAND_TIMEOUT):
    """
    Execute a shell command and return the output.

    Args:
        command: The shell command to execute
        timeout: Maximum execution time in seconds

    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    if not command or not command.strip():
        return False, "", "Empty command"

    # Limit command length for safety
    if len(command) > MAX_COMMAND_LENGTH:
        return False, "", f"Command too long (max {MAX_COMMAND_LENGTH} characters)"

    # Basic safety checks - block obviously dangerous commands
    dangerous_patterns = [
        r'\brm\s+-rf\s+/',  # rm -rf /
        r'\bdd\s+if=',       # dd command (can overwrite disks)
        r':\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}',  # fork bomb
        r'>\s*/dev/',        # redirecting to /dev (could be dangerous)
    ]

    command_lower = command.lower().strip()
    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            return False, "", f"Command blocked for safety reasons: {command}"

    try:
        log.debug(f"Executing command: {command}")

        # Use shell=True to support pipes, redirects, etc.
        # We've already done basic safety checks above
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None  # Use current working directory
        )

        stdout = result.stdout[:MAX_COMMAND_OUTPUT]
        stderr = result.stderr[:MAX_COMMAND_OUTPUT]

        # Truncate if too long
        if len(result.stdout) > MAX_COMMAND_OUTPUT:
            stdout += f"\n... (output truncated, showing first {MAX_COMMAND_OUTPUT} characters)"
        if len(result.stderr) > MAX_COMMAND_OUTPUT:
            stderr += f"\n... (error output truncated, showing first {MAX_COMMAND_OUTPUT} characters)"

        success = result.returncode == 0
        output = stdout + (f"\n[stderr]\n{stderr}" if stderr else "")

        log.debug(f"Command exit code: {result.returncode}")
        return success, output, stderr if not success else ""

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        log.error(error_msg)
        return False, "", error_msg
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        log.error(error_msg)
        return False, "", error_msg


def extract_commands(text):
    """
    Extract commands from text using <execute>...</execute> markers.

    Args:
        text: The text to search for commands

    Returns:
        list: List of (command, start_pos, end_pos) tuples
    """
    # Pattern to match <execute>command</execute>
    pattern = r'<execute>(.*?)</execute>'
    commands = []

    for match in re.finditer(pattern, text, re.DOTALL):
        command = match.group(1).strip()
        if command:
            commands.append((command, match.start(), match.end()))

    return commands


def remove_command_markers(text):
    """
    Remove <execute>...</execute> markers from text, keeping the command text.

    Args:
        text: The text to process

    Returns:
        str: Text with markers removed
    """
    # Replace <execute>command</execute> with just the command
    pattern = r'<execute>(.*?)</execute>'
    return re.sub(pattern, r'\1', text, flags=re.DOTALL)


def sanitize_vim_commands(text):
    """
    Filter out special vim command sequences that could interfere with vim operation.

    This prevents sequences like <ESC>, <C-...>, <CR>, etc. from being interpreted
    as vim commands when written to a vim buffer.

    Args:
        text: The text to sanitize

    Returns:
        str: Text with vim command sequences escaped or removed
    """
    if not text:
        return text

    # Simple approach: Only escape specific vim special key sequences that are
    # commonly problematic when written to a buffer

    # Temporarily replace special markers with placeholders
    text = text.replace('<execute>', '\x00EXECUTE_START\x00')
    text = text.replace('</execute>', '\x00EXECUTE_END\x00')
    text = text.replace('<PROMPT>', '\x00PROMPT_MARKER\x00')
    text = text.replace('<EOT>', '\x00EOT_MARKER\x00')

    # List of specific vim special keys to escape (case-insensitive)
    # These are sequences that vim interprets as special keys
    vim_keys = [
        'ESC', 'Esc', 'esc',
        'CR', 'Enter', 'Return',
        'Tab', 'Space', 'BS', 'Del', 'Insert',
        'Home', 'End', 'PageUp', 'PageDown',
        'Up', 'Down', 'Left', 'Right',
    ]

    # Escape specific vim key sequences: <ESC>, <CR>, etc.
    for key in vim_keys:
        text = text.replace(f'<{key}>', f'\\<{key}>')

    # Also escape Control sequences: <C-x>, <M-x>, <A-x>, <S-x>
    # Use regex for these patterns
    text = re.sub(r'<(C-\w+)>', r'\\<\1>', text, flags=re.IGNORECASE)
    text = re.sub(r'<(M-\w+)>', r'\\<\1>', text, flags=re.IGNORECASE)
    text = re.sub(r'<(A-\w+)>', r'\\<\1>', text, flags=re.IGNORECASE)
    text = re.sub(r'<(S-\w+)>', r'\\<\1>', text, flags=re.IGNORECASE)
    text = re.sub(r'<(F\d+)>', r'\\<\1>', text, flags=re.IGNORECASE)

    # Restore special markers
    text = text.replace('\x00EXECUTE_START\x00', '<execute>')
    text = text.replace('\x00EXECUTE_END\x00', '</execute>')
    text = text.replace('\x00PROMPT_MARKER\x00', '<PROMPT>')
    text = text.replace('\x00EOT_MARKER\x00', '<EOT>')

    return text

async def stream_chat_message_ollama(messages, endpoint, model, options, timeout):
    """Stream chat responses from Ollama API.

    Returns:
        str: The complete assistant message
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Host": endpoint.split("//")[1].split("/")[0],
    }

    data = {
        "model": model,
        "messages": messages,
        "raw": True,
        "options": options,
    }
    log.debug("request: " + json.dumps(data, indent=4))

    assistant_message = ""

    try:
        timeout_config = httpx.Timeout(
            connect=DEFAULT_TIMEOUT,
            read=DEFAULT_TIMEOUT,
            write=DEFAULT_TIMEOUT,
            pool=DEFAULT_TIMEOUT,
        )
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream("POST", endpoint, headers=headers, json=data) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            message = json.loads(line)
                            if "message" in message and "content" in message["message"]:
                                content = message["message"]["content"]
                                assistant_message += content
                                # Sanitize vim commands before printing to buffer
                                sanitized_content = sanitize_vim_commands(content)
                                print(sanitized_content, end="", flush=True)

                                # If <EOT> is detected, stop processing
                                if "<EOT>" in content:
                                    break
                            # Stop if response contains an indication of completion
                            if message.get("done", False):
                                # Don't print <EOT> here as it might interfere with vim
                                # The plugin should handle detecting the end of stream
                                break
                else:
                    await response.aread()
                    raise Exception(f"Error: {response.status_code} - {response.text}")
    except httpx.ReadTimeout:
        error_msg = "\n[ERROR] Read timeout occurred. Please try again."
        print(error_msg, flush=True)
        log.error("Read timeout occurred.")
        # Return empty to signal error - don't continue processing
        return ""
    except asyncio.CancelledError:
        log.info("Task was cancelled.")
        raise
    except Exception as e:
        error_msg = sanitize_vim_commands(f"\n[ERROR] An error occurred: {str(e)}")
        print(error_msg, flush=True)
        log.error(f"An error occurred: {str(e)}")
        # Return empty to signal error - don't continue processing
        return ""

    # Add the assistant's message to the conversation history
    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message.strip()})

    return assistant_message.strip()


async def stream_chat_message_openai(messages, endpoint, model, options, credentialname):
    """Stream chat responses from OpenAI API.

    Returns:
        str: The complete assistant message
    """
    if AsyncOpenAI is None:
        raise ImportError("OpenAI package not found. Please install via 'pip install openai'.")

    log.debug('Using OpenAI completion endpoint')
    cred = OllamaCredentials()
    api_key = cred.GetApiKey('openai', credentialname)
    # don't trace API keys in production, just a development helper
    #log.debug(f'api_key={api_key}')

    if endpoint:
        log.info('Using OpenAI endpoint '+endpoint)
        client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    else:
        log.info('Using official OpenAI endpoint')
        client = AsyncOpenAI(api_key=api_key)
    assistant_message = ""

    temperature = options.get('temperature', DEFAULT_TEMPERATURE)
    max_tokens = options.get('max_tokens', DEFAULT_MAX_TOKENS)
    top_p = options.get('top_p', 1.0)

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                assistant_message += token
                # Sanitize vim commands before printing to buffer
                sanitized_token = sanitize_vim_commands(token)
                print(sanitized_token, end="", flush=True)

        # Print end marker (sanitized to avoid vim command interference)
        print(sanitize_vim_commands("<EOT>"), flush=True)

    except Exception as e:
        error_msg = sanitize_vim_commands(f"\n[ERROR] An error occurred: {str(e)}")
        print(error_msg, flush=True)
        log.error(f"Error in OpenAI stream: {str(e)}")
        # Return empty to signal error - don't continue processing
        return ""

    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message.strip()})

    return assistant_message.strip()

async def main(provider, endpoint, model, options, systemprompt, timeout, credentialname, enable_commands=False, confirm_commands=False):
    conversation_history = []
    log.debug("endpoint: " + str(endpoint))

    multiline_input = False
    multiline_message = []

    # Ensure systemprompt is initialized
    if not systemprompt:
        if provider == "ollama":
            systemprompt = f"Today's date is {datetime.date.today().isoformat()}"
        else:
            systemprompt = ""
    else:
        if provider == "ollama":
            # Let Ollama know the current date
            systemprompt = f"Today's date is {datetime.date.today().isoformat()}\n\n{systemprompt}"

    # Add formatting constraint to system prompt
    systemprompt = f"{systemprompt}\n\nIMPORTANT: Format all responses to be readable from the terminal. Each line must not exceed 75 characters. Break long lines naturally at word boundaries. Do not mention this formatting requirement or terminal readability in your responses."

    # Add command execution instructions if enabled
    if enable_commands:
        command_instructions = """
You have the ability to execute shell commands when needed to help answer questions or perform tasks.

To execute a command, wrap it in <execute>...</execute> tags. For example:
- To list files: <execute>ls -la</execute>
- To check the date: <execute>date</execute>
- To get system info: <execute>uname -a</execute>

IMPORTANT RULES:
1. Only execute commands that are safe and necessary to answer the user's question
2. Execute commands one at a time
3. After executing a command, analyze the output and provide a helpful response
4. If a command fails, explain what went wrong and suggest alternatives
5. Never execute destructive commands like 'rm -rf', 'dd', or commands that modify system files without explicit user request
6. Always explain what you're doing and why

When you execute a command, the output will be provided to you automatically. Continue your response after seeing the command output.
"""
        systemprompt = f"{systemprompt}\n\n{command_instructions}"

    conversation_history.append({"role": "system", "content": systemprompt})

    while True:
        try:
            user_message = input("").strip()

            if multiline_input:
                if user_message == '"""':
                    multiline_input = False
                    complete_message = "\n".join(multiline_message)
                    conversation_history.append({"role": "user", "content": complete_message})
                    multiline_message = []

                    # Handle command execution loop
                    await process_message_with_commands(
                        provider, endpoint, model, options, timeout, credentialname,
                        conversation_history, enable_commands, confirm_commands
                    )
                else:
                    multiline_message.append(user_message)
            else:
                if user_message == '"""':
                    multiline_input = True
                    multiline_message = []
                elif user_message.lower() in ['exit', 'quit', '/bye', ':q']:
                    print("Exiting the chat.")
                    exit(0)
                else:
                    conversation_history.append({"role": "user", "content": user_message})
                    # Handle command execution loop
                    await process_message_with_commands(
                        provider, endpoint, model, options, timeout, credentialname,
                        conversation_history, enable_commands, confirm_commands
                    )

        except KeyboardInterrupt:
            print("\nStreaming interrupted. Showing prompt again...")
            if "task" in locals():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def confirm_command_execution(command):
    """
    Prompt user for confirmation before executing a command.

    Args:
        command: The command to confirm

    Returns:
        bool: True if user confirmed, False otherwise
    """
    # Use <PROMPT> marker to signal vim to enter insert mode
    print("\n[CONFIRM] Type 'y' or 'yes' to execute, 'n' or 'no' to skip: <PROMPT>", flush=True)

    try:
        # Read user input from stdin
        # Note: This will block until user enters a response
        response = input().strip().lower()

        # Check for confirmation
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("[CONFIRM] Execution cancelled.", flush=True)
            return False
        else:
            # Invalid response, default to no for safety
            sanitized_response = sanitize_vim_commands(response)
            print(f"[CONFIRM] Invalid response '{sanitized_response}'. Execution cancelled.", flush=True)
            return False
    except (EOFError, KeyboardInterrupt):
        # Handle Ctrl-C or EOF
        print("\n[CONFIRM] Execution cancelled.", flush=True)
        return False


async def process_message_with_commands(provider, endpoint, model, options, timeout, credentialname,
                                       conversation_history, enable_commands,
                                       confirm_commands=False, max_iterations=10):
    """
    Process a message with ReAct-style command execution loop.

    This function:
    1. Sends the message to the LLM
    2. Detects any commands in the response
    3. Prompts for confirmation if enabled
    4. Executes commands and feeds results back
    5. Repeats until no more commands are found or max iterations reached

    Args:
        confirm_commands: If True, prompt user for confirmation before executing each command
    """
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Log that we're about to query the LLM (for debugging hung issues)
        log.debug(f"Starting iteration {iteration}, querying LLM...")

        # Get response from LLM
        if provider == "ollama":
            assistant_message = await stream_chat_message_ollama(
                conversation_history, endpoint, model, options, timeout
            )
        else:
            assistant_message = await stream_chat_message_openai(
                conversation_history, endpoint, model, options, credentialname
            )

        # Check if we got an empty response (error occurred)
        if not assistant_message or not assistant_message.strip():
            log.warning("Received empty response from LLM, stopping command loop")
            print("\n[WARNING] No response from LLM. Stopping command execution.", flush=True)
            break

        # Check if command execution is enabled and if there are commands
        if not enable_commands:
            break

        commands = extract_commands(assistant_message)
        if not commands:
            # No commands found, we're done
            break

        # Execute each command and prepare feedback
        command_results = []
        for command, start_pos, end_pos in commands:
            # Ask for confirmation if enabled
            if confirm_commands:
                if not confirm_command_execution(command):
                    # User declined, skip this command
                    command_results.append(f"Command: {command}\nResult: [SKIPPED - User declined execution]")
                    print("<PROMPT>", flush=True)
                    return

            success, output, error = execute_command(command, timeout=DEFAULT_COMMAND_TIMEOUT)

            if success:
                result_text = f"Command executed successfully:\n{output}"
                command_results.append(f"Command: {command}\nResult: {output}")
            else:
                result_text = f"Command failed: {error}\nOutput: {output}" if output else f"Command failed: {error}"
                command_results.append(f"Command: {command}\nError: {error}\nOutput: {output}")

            # Sanitize vim commands before printing to buffer
            sanitized_result = sanitize_vim_commands(result_text)
            print(sanitized_result, flush=True)

        # If we executed commands (or skipped some), feed the results back to the LLM
        if command_results:
            # Create a user message with the command results
            results_message = "Command execution results:\n\n" + "\n\n---\n\n".join(command_results)
            conversation_history.append({"role": "user", "content": results_message})

            # Continue the conversation - the LLM should analyze the results
            # We'll get another response in the next iteration
            print("\n[Analyzing command results...]\n", flush=True)
            log.debug(f"Continuing to iteration {iteration + 1} to analyze command results")

    # Use <PROMPT> marker to signal vim to enter insert mode
    print("\n[DONE]<PROMPT>", flush=True)
    if iteration >= max_iterations:
        print(f"\n[Warning: Reached maximum command execution iterations ({max_iterations})]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with an Ollama or OpenAI LLM.")
    parser.add_argument("-p", "--provider", type=str, default=DEFAULT_PROVIDER,
                        choices=["ollama", "openai"],
                        help="LLM provider: 'ollama' (default) or 'openai'")
    parser.add_argument("-m", "--model", type=str, default=None, help="Specify the model name to use.")
    parser.add_argument("-u", "--url", type=str, default=None,
                        help="Base endpoint URL.")
    parser.add_argument("-o", "--options", type=str, default=DEFAULT_OPTIONS,
                        help="Ollama REST API options.")
    parser.add_argument("-s", "--system-prompt", type=str, default="", help="Specify system prompt.")
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds.")
    parser.add_argument("-l", "--log-level", type=int, default=OllamaLogger.ERROR, help="Log level.")
    parser.add_argument("-f", "--log-filename", type=str, default="chat.log", help="Log filename.")
    parser.add_argument("-d", "--log-dir", type=str, default="/tmp/logs", help="Log file directory.")
    parser.add_argument('-k', '--keyname', default=None,
                        help="Credential name to lookup API key and password store")
    parser.add_argument('-c', '--enable-commands', action='store_true', default=False,
                        help="Enable shell command execution capability.")
    parser.add_argument('--confirm-commands', action='store_true', default=False,
                        help="Prompt for confirmation before executing each command.")
    args = parser.parse_args()

    log = OllamaLogger(args.log_dir, args.log_filename)
    log.setLevel(args.log_level)

    # Parse options JSON
    try:
        options = json.loads(args.options)
    except json.JSONDecodeError:
        options = json.loads(DEFAULT_OPTIONS)

    # Choose model defaults
    if args.provider == "openai":
        model = args.model or DEFAULT_OPENAI_MODEL
        endpoint = args.url or None
    else:
        model = args.model or DEFAULT_MODEL
        endpoint = args.url or DEFAULT_HOST
        endpoint = endpoint + "/api/chat"

    try:
        while True:
            try:
                asyncio.run(main(args.provider, endpoint, model, options, args.system_prompt, args.timeout, args.keyname, args.enable_commands, args.confirm_commands))
            except KeyboardInterrupt:
                print("Canceled.")
                break
    except Exception as e:
        # Print only the root cause message, not the full traceback
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nExiting the chat. (outer)")
