#!/usr/bin/env python3
"""
N8N Workflow Health Checker
Checks workflow health and suggests improvements using Ollama AI
"""

import sys
import os
import json
import copy
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Track if we've already shown Ollama connection warnings
_ollama_warning_shown = False
# Store user's fallback service choice
_fallback_service_choice = None


def load_env():
    """Load environment variables from .env file"""
    load_dotenv()

    # Required n8n credentials
    required_vars = ['N8N_URL', 'N8N_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(f"[red]Error: Missing required environment variables: {', '.join(missing)}[/red]")
        console.print("[yellow]Copy .env.example to .env and fill in your credentials[/yellow]")
        sys.exit(1)

    # At least one AI service required (Ollama, Groq, or Gemini)
    has_ollama = os.getenv('OLLAMA_URL') and os.getenv('OLLAMA_MODEL')
    has_groq = os.getenv('GROQ_API_KEY')
    has_gemini = os.getenv('GEMINI_API_KEY')

    if not has_ollama and not has_groq and not has_gemini:
        console.print("[red]Error: At least one AI service must be configured:[/red]")
        console.print("[yellow]  - Ollama (OLLAMA_URL + OLLAMA_MODEL), or[/yellow]")
        console.print("[yellow]  - Groq (GROQ_API_KEY), or[/yellow]")
        console.print("[yellow]  - Gemini (GEMINI_API_KEY)[/yellow]")
        sys.exit(1)


def fetch_workflows(workflow_id=None):
    """Fetch workflows from n8n API"""
    base_url = os.getenv('N8N_URL').rstrip('/')
    url = f"{base_url}/api/v1/workflows"
    if workflow_id:
        url += f"/{workflow_id}"

    headers = {"X-N8N-API-KEY": os.getenv('N8N_API_KEY')}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [data] if workflow_id else data.get('data', [])
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON response: {e}[/red]")
        console.print(f"[yellow]Response status: {response.status_code}[/yellow]")
        console.print(f"[yellow]Response text: {response.text[:200]}[/yellow]")
        return []
    except requests.RequestException as e:
        console.print(f"[red]Error fetching workflows: {e}[/red]")
        return []


def update_workflow(workflow_id, workflow_data):
    """Update a workflow via n8n API"""
    base_url = os.getenv('N8N_URL').rstrip('/')
    url = f"{base_url}/api/v1/workflows/{workflow_id}"
    headers = {
        "X-N8N-API-KEY": os.getenv('N8N_API_KEY'),
        "Content-Type": "application/json"
    }

    # Clean workflow data - remove read-only fields that n8n doesn't accept
    clean_data = copy.deepcopy(workflow_data)
    read_only_fields = ['id', 'createdAt', 'updatedAt', 'versionId', 'meta']
    for field in read_only_fields:
        clean_data.pop(field, None)

    try:
        response = requests.put(url, headers=headers, json=clean_data, timeout=30)
        response.raise_for_status()
        return {'success': True, 'data': response.json()}
    except requests.RequestException as e:
        # Get detailed error message from n8n
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                if 'message' in error_detail:
                    error_msg = f"{error_msg} - {error_detail['message']}"
                elif isinstance(error_detail, dict):
                    error_msg = f"{error_msg} - {json.dumps(error_detail)}"
            except:
                error_msg = f"{error_msg} - {e.response.text[:200]}"
        return {'success': False, 'error': error_msg}


def fetch_executions(workflow_id, limit=20):
    """Fetch recent executions for a workflow"""
    base_url = os.getenv('N8N_URL').rstrip('/')
    url = f"{base_url}/api/v1/executions"
    headers = {"X-N8N-API-KEY": os.getenv('N8N_API_KEY')}
    params = {"workflowId": workflow_id, "limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except (json.JSONDecodeError, requests.RequestException):
        return []


def fetch_execution_details(execution_id):
    """Fetch full execution details including error messages"""
    base_url = os.getenv('N8N_URL').rstrip('/')
    url = f"{base_url}/api/v1/executions/{execution_id}"
    headers = {"X-N8N-API-KEY": os.getenv('N8N_API_KEY')}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except (json.JSONDecodeError, requests.RequestException):
        return None


def analyze_workflow_structure(workflow):
    """Analyze workflow structure for complexity and error handling"""
    nodes = workflow.get('nodes', [])
    connections = workflow.get('connections', {})
    
    node_types = {}
    for node in nodes:
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Check for error handling (Error Trigger node or IF nodes after critical operations)
    has_error_handling = any(node.get('type') == 'n8n-nodes-base.errorTrigger' for node in nodes)
    
    # Check for retry logic (nodes with retryOnFail or retry settings)
    has_retry_logic = any(
        node.get('parameters', {}).get('retryOnFail') or 
        node.get('parameters', {}).get('retry') 
        for node in nodes
    )
    
    return {
        'node_count': len(nodes),
        'node_types': node_types,
        'has_error_handling': has_error_handling,
        'has_retry_logic': has_retry_logic
    }


def extract_error_messages(executions):
    """Extract unique error messages from failed executions"""
    error_messages = []
    failed_execution_ids = []
    
    for execution in executions:
        # Check if execution failed
        finished = execution.get('finished', False)
        exec_data = execution.get('data', {})
        result_data = exec_data.get('resultData', {})
        has_error = bool(result_data.get('error'))
        
        # Execution failed if it finished but has error, or if it didn't finish
        if finished and has_error:
            failed_execution_ids.append(execution.get('id'))
        elif not finished:
            # Execution didn't complete - might be in progress or failed
            failed_execution_ids.append(execution.get('id'))
    
    # Fetch full details for failed executions to get error messages
    unique_errors = set()
    for exec_id in failed_execution_ids[:10]:  # Limit to 10 to avoid too many API calls
        details = fetch_execution_details(exec_id)
        if details:
            # Extract error messages from execution data
            data = details.get('data', {})
            result_data = data.get('resultData', {})
            
            # Check for errors in resultData
            if result_data.get('error'):
                error = result_data['error']
                error_msg = error.get('message', str(error))
                unique_errors.add(error_msg)
            
            # Check for errors in node execution data
            for node_name, node_data in result_data.get('runData', {}).items():
                for output in node_data.get('output', []):
                    for item in output:
                        if isinstance(item, dict) and 'error' in item:
                            error_msg = item['error'].get('message', str(item['error']))
                            unique_errors.add(error_msg)
    
    return list(unique_errors)


def calculate_health(workflow, executions):
    """Calculate workflow health metrics"""
    total = len(executions)
    if total == 0:
        return {
            'success_rate': 0,
            'success_count': 0,
            'failed_count': 0,
            'total_count': 0,
            'last_run': None,
            'last_success': None,
            'never_succeeded': True,
            'error_messages': []
        }

    successful = 0
    failed = 0
    last_run = None
    last_success = None

    for execution in executions:
        finished = execution.get('finished', False)
        stopped_at = execution.get('stoppedAt')
        started_at = execution.get('startedAt')
        
        # Check for errors in execution data
        has_error = False
        exec_data = execution.get('data', {})
        result_data = exec_data.get('resultData', {})
        if result_data.get('error'):
            has_error = True
        
        # Determine if execution was successful
        # Success: finished is True and no error in resultData
        # Note: stoppedAt is the timestamp when execution stopped (success or failure)
        is_success = finished and not has_error
        
        if is_success:
            successful += 1
            if stopped_at and (not last_success or stopped_at > last_success):
                last_success = stopped_at
        else:
            failed += 1
        
        # Track last run (most recent stoppedAt, or startedAt if no stoppedAt)
        if stopped_at and (not last_run or stopped_at > last_run):
            last_run = stopped_at
        elif started_at and (not last_run or started_at > last_run):
            last_run = started_at

    # Extract error messages from failed executions
    error_messages = extract_error_messages(executions)

    return {
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'success_count': successful,
        'failed_count': failed,
        'total_count': total,
        'last_run': last_run,
        'last_success': last_success,
        'never_succeeded': successful == 0 and total > 0,
        'error_messages': error_messages
    }


def time_ago(timestamp_str):
    """Convert timestamp to human-readable time ago"""
    if not timestamp_str:
        return "Never"

    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - timestamp

        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        minutes = (diff.seconds % 3600) // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        return "Just now"
    except Exception:
        return "Unknown"


def prompt_fallback_service():
    """Prompt user to choose fallback AI service when Ollama fails"""
    global _fallback_service_choice
    
    # Return cached choice if available
    if _fallback_service_choice:
        return _fallback_service_choice
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    available_services = []
    if groq_api_key:
        available_services.append(('groq', 'Groq'))
    if gemini_api_key:
        available_services.append(('gemini', 'Gemini'))
    
    if not available_services:
        return None
    
    if len(available_services) == 1:
        # Only one option, use it automatically
        _fallback_service_choice = available_services[0][0]
        return _fallback_service_choice
    
    # Multiple options, prompt user
    console.print()
    console.print("[yellow]Ollama is not available. Choose a fallback AI service:[/yellow]")
    for i, (key, name) in enumerate(available_services, 1):
        console.print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input("\n[bold green]Enter choice (1-{}):[/bold green] ".format(len(available_services))).strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_services):
                _fallback_service_choice = available_services[choice_num - 1][0]
                console.print(f"[green]Using {available_services[choice_num - 1][1]} as fallback[/green]")
                console.print()
                return _fallback_service_choice
            else:
                console.print("[red]Invalid choice. Please enter a number between 1 and {}[/red]".format(len(available_services)))
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled. Using first available service.[/yellow]")
            _fallback_service_choice = available_services[0][0]
            return _fallback_service_choice


def call_gemini_api(prompt):
    """Call Google Gemini API"""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    # Default to gemini-2.5-flash (latest fast model) or use gemini-2.5-pro for better quality
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    if not gemini_api_key:
        return None
    
    try:
        # Try v1beta first
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # Extract text from Gemini response
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if parts and 'text' in parts[0]:
                return parts[0]['text']
        
        return None
    except requests.exceptions.HTTPError as e:
        # If v1beta fails with 404, try v1 API as fallback
        if e.response and e.response.status_code == 404:
            try:
                url = f"https://generativelanguage.googleapis.com/v1/models/{gemini_model}:generateContent?key={gemini_api_key}"
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                # Extract text from Gemini response
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']
            except requests.RequestException:
                pass  # Fall through to error message
        
        # Only show error if both attempts failed
        console.print(f"[red]Gemini API HTTP error: {e}[/red]", style="dim")
        if hasattr(e.response, 'text'):
            error_text = e.response.text[:200]
            console.print(f"[red]Response: {error_text}[/red]", style="dim")
            # Suggest valid model names if model not found
            if 'not found' in error_text.lower() or 'not supported' in error_text.lower():
                console.print("[yellow]💡 Tip: Try setting GEMINI_MODEL to 'gemini-1.5-flash' or 'gemini-1.5-pro'[/yellow]", style="dim")
        return None
    except requests.RequestException as e:
        console.print(f"[red]Gemini API request failed: {e}[/red]", style="dim")
        return None


def ask_ollama(workflow, health, structure):
    """Ask Ollama (or fallback to Gemini/Groq) for comprehensive workflow analysis"""
    global _ollama_warning_shown
    active_status = "active" if workflow.get('active') else "inactive"
    
    # Build failure analysis section
    failure_analysis = ""
    if health['failed_count'] > 0 and health['error_messages']:
        failure_analysis = "\n\nFAILURE ANALYSIS:\n\nRecent error messages:\n"
        for i, error in enumerate(health['error_messages'][:10], 1):
            failure_analysis += f"{i}. {error}\n"
    elif health['failed_count'] > 0:
        failure_analysis = "\n\nFAILURE ANALYSIS:\n\nExecutions are failing but no detailed error messages were captured."
    
    # Build workflow structure section
    node_types_list = ", ".join([f"{k} ({v})" for k, v in structure['node_types'].items()])
    
    prompt = f"""You are an n8n automation expert analyzing a workflow's health and reliability.

Workflow: {workflow['name']}
Status: {active_status}

EXECUTION HISTORY:

- Total executions checked: {health['total_count']}
- Success rate: {health['success_rate']:.1f}%
- Successful runs: {health['success_count']}
- Failed runs: {health['failed_count']}
- Last successful run: {time_ago(health['last_success'])}
- Last run: {time_ago(health['last_run'])}
{failure_analysis}
WORKFLOW STRUCTURE:

- Total nodes: {structure['node_count']}
- Node types: {node_types_list}
- Has error handling: {'Yes' if structure['has_error_handling'] else 'No'} - check for error workflow or IF nodes after critical operations
- Has retry logic: {'Yes' if structure['has_retry_logic'] else 'No'}

WORKFLOW JSON:

{json.dumps(workflow, indent=2)}

Based on this complete picture, provide:

1. ROOT CAUSE: What's actually wrong with this workflow? Be specific.
   - If it's never run successfully, is it incomplete or misconfigured?
   - If it's failing consistently, what's the likely issue?
   - If success rate is low, what pattern do you see in the errors?

2. QUICK FIX: ONE specific action to improve reliability (5-30 min to implement)

3. IMPLEMENTATION: Provide concrete, implementable changes. Choose ONE approach:
   - ADD_RETRY: Add retry logic to a specific node. Specify node name and retry settings.
   - ADD_ERROR_HANDLING: Add an error workflow or error trigger node.
   - UPDATE_NODE: Update a node's parameters (e.g., timeout, credentials, settings).
   - ADD_NODE: Add a new node to the workflow (e.g., IF node for error handling).
   - FIX_SETTINGS: Update workflow-level settings.

   Format as JSON:
   {{"action": "ADD_RETRY|ADD_ERROR_HANDLING|UPDATE_NODE|ADD_NODE|FIX_SETTINGS", "details": {{"node_name": "...", "changes": {{...}}}}}}

4. STATUS: Is this workflow:
   - HEALTHY: Working as intended
   - NEEDS ATTENTION: Working but could fail
   - BROKEN: Not working, needs immediate fix
   - INCOMPLETE: Never completed, needs finishing

Be direct and practical. If you can't tell what's wrong from the data, say so.

Format your response as:
ROOT CAUSE:
[your analysis]

QUICK FIX:
[your suggestion]

IMPLEMENTATION:
```json
[your implementation JSON]
```

STATUS:
[HEALTHY/NEEDS ATTENTION/BROKEN/INCOMPLETE]"""

    # Try Ollama first
    ollama_url = os.getenv('OLLAMA_URL')
    ollama_model = os.getenv('OLLAMA_MODEL')
    
    # Default to localhost if OLLAMA_URL not set
    if not ollama_url:
        ollama_url = 'http://localhost:11434'
        if not _ollama_warning_shown:
            console.print("[dim]OLLAMA_URL not set, using default: http://localhost:11434[/dim]")
    
    if not ollama_model:
        if not _ollama_warning_shown:
            console.print("[yellow]⚠️  OLLAMA_MODEL not set in environment variables[/yellow]")
            console.print("[yellow]   Add OLLAMA_MODEL to your .env file (e.g., OLLAMA_MODEL=llama3.2)[/yellow]")
            console.print()
    
    if ollama_url and ollama_model:
        try:
            base_url = ollama_url.rstrip('/')
            url = f"{base_url}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            response_text = response.json().get('response', 'Unable to generate suggestion')
            return {
                'response': response_text,
                'service': 'Ollama',
                'model': ollama_model
            }
        except requests.exceptions.ConnectionError as e:
            if not _ollama_warning_shown:
                console.print(f"[yellow]⚠️  Ollama connection failed: {e}[/yellow]")
                console.print(f"[yellow]   Trying to connect to: {ollama_url}[/yellow]")
                console.print("[yellow]   Make sure Ollama is running: ollama serve[/yellow]")
                _ollama_warning_shown = True
        except requests.exceptions.Timeout:
            if not _ollama_warning_shown:
                console.print("[yellow]⚠️  Ollama request timed out.[/yellow]")
                _ollama_warning_shown = True
        except requests.exceptions.HTTPError as e:
            if not _ollama_warning_shown:
                console.print(f"[yellow]⚠️  Ollama HTTP error: {e}[/yellow]")
                console.print(f"[yellow]   Check if model '{ollama_model}' exists: ollama list[/yellow]")
                _ollama_warning_shown = True
        except requests.RequestException as e:
            if not _ollama_warning_shown:
                console.print(f"[yellow]⚠️  Ollama request failed: {e}[/yellow]")
                _ollama_warning_shown = True

    # Ollama failed or not configured - prompt for fallback service
    fallback_service = prompt_fallback_service()
    
    if not fallback_service:
        # No fallback services available
        return {
            'response': "Unable to connect to AI service. Ollama failed and no fallback services (Groq/Gemini) are configured.",
            'service': 'None',
            'model': 'N/A'
        }
    
    if fallback_service == 'groq':
        # Use Groq
        groq_api_key = os.getenv('GROQ_API_KEY')
        groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        if groq_api_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": groq_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                response_text = response.json()['choices'][0]['message']['content']
                return {
                    'response': response_text,
                    'service': 'Groq',
                    'model': groq_model
                }
            except requests.RequestException as e:
                console.print(f"[red]Groq API error: {e}[/red]")
    
    elif fallback_service == 'gemini':
        # Use Gemini
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        response_text = call_gemini_api(prompt)
        if response_text:
            return {
                'response': response_text,
                'service': 'Gemini',
                'model': gemini_model
            }
        else:
            console.print("[red]Gemini API call failed[/red]")

    # All services failed
    return {
        'response': "Unable to connect to AI service (tried Ollama and fallback)",
        'service': 'None',
        'model': 'N/A'
    }


def parse_ai_response(response_text):
    """Parse AI response into structured components"""
    result = {
        'root_cause': '',
        'quick_fix': '',
        'implementation': None,
        'status': ''
    }

    # Extract ROOT CAUSE
    if 'ROOT CAUSE:' in response_text:
        parts = response_text.split('ROOT CAUSE:')
        if len(parts) > 1:
            remaining = parts[1]
            if 'QUICK FIX:' in remaining:
                result['root_cause'] = remaining.split('QUICK FIX:')[0].strip()
            else:
                result['root_cause'] = remaining.strip()

    # Extract QUICK FIX
    if 'QUICK FIX:' in response_text:
        parts = response_text.split('QUICK FIX:')
        if len(parts) > 1:
            remaining = parts[1]
            if 'IMPLEMENTATION:' in remaining:
                result['quick_fix'] = remaining.split('IMPLEMENTATION:')[0].strip()
            elif 'STATUS:' in remaining:
                result['quick_fix'] = remaining.split('STATUS:')[0].strip()
            else:
                result['quick_fix'] = remaining.strip()

    # Extract IMPLEMENTATION (JSON)
    if 'IMPLEMENTATION:' in response_text or '```json' in response_text:
        try:
            # Try to find JSON code block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end > json_start:
                    json_str = response_text[json_start:json_end].strip()
                    result['implementation'] = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, implementation will remain None
            pass

    # Extract STATUS
    if 'STATUS:' in response_text:
        parts = response_text.split('STATUS:')
        if len(parts) > 1:
            result['status'] = parts[1].strip().split('\n')[0].strip()

    # If parsing failed, return raw response
    if not any([result['root_cause'], result['quick_fix'], result['status']]):
        result['root_cause'] = response_text

    return result


def apply_fix_to_workflow(workflow, implementation):
    """Apply AI-suggested fix to workflow JSON"""
    if not implementation or 'action' not in implementation:
        return {'success': False, 'error': 'No implementation provided'}

    action = implementation.get('action', '')
    details = implementation.get('details', {})

    # Make a deep copy to avoid modifying the original
    try:
        updated_workflow = copy.deepcopy(workflow)
        nodes = updated_workflow.get('nodes', [])
    except Exception as e:
        return {'success': False, 'error': f'Failed to copy workflow: {str(e)}'}

    if action == 'ADD_RETRY':
        # Add retry logic to a specific node
        node_name = details.get('node_name', '')
        if not node_name:
            return {'success': False, 'error': 'Node name not specified'}

        retry_settings = details.get('changes', {})
        node_found = False

        for node in nodes:
            if node.get('name') == node_name:
                node_found = True
                # Add retry settings to node
                if 'retryOnFail' not in node:
                    node['retryOnFail'] = True
                if 'maxTries' not in node:
                    node['maxTries'] = retry_settings.get('maxTries', 3)
                if 'waitBetweenTries' not in node:
                    node['waitBetweenTries'] = retry_settings.get('waitBetweenTries', 1000)
                return {'success': True, 'workflow': updated_workflow}

        if not node_found:
            return {'success': False, 'error': f'Node "{node_name}" not found in workflow'}

    elif action == 'UPDATE_NODE':
        # Update node parameters
        node_name = details.get('node_name', '')
        if not node_name:
            return {'success': False, 'error': 'Node name not specified'}

        changes = details.get('changes', {})
        node_found = False

        for node in nodes:
            if node.get('name') == node_name:
                node_found = True
                # Update parameters
                if 'parameters' not in node:
                    node['parameters'] = {}
                for key, value in changes.items():
                    node['parameters'][key] = value
                return {'success': True, 'workflow': updated_workflow}

        if not node_found:
            return {'success': False, 'error': f'Node "{node_name}" not found in workflow'}

    elif action == 'FIX_SETTINGS':
        # Update workflow settings
        settings = details.get('changes', {})
        if not settings:
            return {'success': False, 'error': 'No settings changes specified'}

        if 'settings' not in updated_workflow:
            updated_workflow['settings'] = {}

        for key, value in settings.items():
            updated_workflow['settings'][key] = value
        return {'success': True, 'workflow': updated_workflow}

    elif action == 'ADD_ERROR_HANDLING':
        # Add error trigger node (simplified - would need proper positioning)
        # Find a good position (to the right of existing nodes)
        max_x = 0
        for node in nodes:
            pos = node.get('position', [0, 0])
            if len(pos) >= 1 and pos[0] > max_x:
                max_x = pos[0]

        error_node = {
            "parameters": {},
            "name": "Error Trigger",
            "type": "n8n-nodes-base.errorTrigger",
            "typeVersion": 1,
            "position": [max_x + 250, 300]
        }
        nodes.append(error_node)
        return {'success': True, 'workflow': updated_workflow}

    elif action == 'ADD_NODE':
        # Add a new node based on details
        new_node = details.get('node', {})
        if not new_node:
            return {'success': False, 'error': 'No node definition provided'}

        nodes.append(new_node)
        return {'success': True, 'workflow': updated_workflow}

    return {'success': False, 'error': f'Unknown action: {action}'}


def prompt_for_fix_approval(workflow, implementation, quick_fix):
    """Prompt user to approve fix implementation"""
    if not implementation:
        return False

    console.print()
    console.print(Panel(
        f"[bold yellow]Fix Available for: {workflow['name']}[/bold yellow]",
        style="yellow"
    ))
    console.print()
    console.print(f"[bold]Suggested Fix:[/bold] {quick_fix}")
    console.print()

    # Display what will change
    action = implementation.get('action', '')
    details = implementation.get('details', {})

    console.print("[bold cyan]Changes to be applied:[/bold cyan]")
    if action == 'ADD_RETRY':
        console.print(f"  • Add retry logic to node: [yellow]{details.get('node_name', 'Unknown')}[/yellow]")
        changes = details.get('changes', {})
        if changes:
            console.print(f"    - Max retries: {changes.get('maxTries', 3)}")
            console.print(f"    - Wait between retries: {changes.get('waitBetweenTries', 1000)}ms")

    elif action == 'UPDATE_NODE':
        console.print(f"  • Update node: [yellow]{details.get('node_name', 'Unknown')}[/yellow]")
        changes = details.get('changes', {})
        if changes:
            for key, value in changes.items():
                console.print(f"    - {key}: {value}")

    elif action == 'FIX_SETTINGS':
        console.print("  • Update workflow settings")
        changes = details.get('changes', {})
        if changes:
            for key, value in changes.items():
                console.print(f"    - {key}: {value}")

    elif action == 'ADD_ERROR_HANDLING':
        console.print("  • Add Error Trigger node for better error handling")

    elif action == 'ADD_NODE':
        console.print(f"  • Add new node to workflow")

    else:
        console.print(f"  • Action: {action}")

    console.print()

    # Prompt for approval
    response = input("[bold green]Apply this fix? (yes/no):[/bold green] ").strip().lower()
    return response in ['yes', 'y']


def get_status_icon(workflow, health, ai_status=""):
    """Get status icon based on workflow health and AI analysis"""
    if not workflow.get('active'):
        return "🔴"

    # Use AI status if available
    if ai_status:
        ai_status_upper = ai_status.upper()
        if 'HEALTHY' in ai_status_upper:
            return "🟢"
        elif 'BROKEN' in ai_status_upper or 'INCOMPLETE' in ai_status_upper:
            return "🔴"
        elif 'NEEDS ATTENTION' in ai_status_upper:
            return "🟡"

    # Fallback to success rate
    if health['success_rate'] >= 90:
        return "🟢"
    elif health['success_rate'] >= 70:
        return "🟡"
    else:
        return "🔴"


def display_workflow_health(workflow, health, ai_analysis):
    """Display health information for a single workflow"""
    # Handle both old format (string) and new format (dict)
    if isinstance(ai_analysis, dict):
        ai_response_text = ai_analysis.get('response', '')
        ai_service = ai_analysis.get('service', 'Unknown')
        ai_model = ai_analysis.get('model', 'Unknown')
    else:
        # Backward compatibility with old string format
        ai_response_text = ai_analysis
        ai_service = 'Unknown'
        ai_model = 'Unknown'
    
    parsed = parse_ai_response(ai_response_text)
    icon = get_status_icon(workflow, health, parsed.get('status', ''))

    output = Text()
    output.append(f"{icon} ", style="bold")
    output.append(workflow['name'], style="bold cyan")
    output.append("\n")

    # Status
    status = "Active" if workflow.get('active') else "Inactive"
    status_color = "green" if workflow.get('active') else "red"
    output.append(f"   Status: ", style="dim")
    output.append(status, style=status_color)
    output.append("\n")

    # Success rate
    if health['total_count'] > 0:
        rate_color = "green" if health['success_rate'] >= 90 else "yellow" if health['success_rate'] >= 70 else "red"
        output.append(f"   Success Rate: ", style="dim")
        output.append(f"{health['success_rate']:.0f}% ({health['success_count']}/{health['total_count']} runs)", style=rate_color)
        
        if health['never_succeeded']:
            output.append(" ⚠️", style="yellow")
            output.append("\n   ", style="dim")
            output.append("Never succeeded: This workflow has never completed successfully", style="yellow")
        
        output.append("\n")

    # Last run
    output.append(f"   Last Run: ", style="dim")
    last_run_time = time_ago(health['last_run'])
    output.append(last_run_time)
    output.append("\n")

    # Root Cause
    if parsed.get('root_cause'):
        output.append("\n   ", style="dim")
        output.append("🔍 ROOT CAUSE:", style="bold yellow")
        output.append("\n   ", style="dim")
        # Split into lines and indent
        root_cause_lines = parsed['root_cause'].split('\n')
        for line in root_cause_lines[:5]:  # Limit to 5 lines
            if line.strip():
                output.append(line.strip(), style="white")
                output.append("\n   ", style="dim")

    # Quick Fix
    if parsed.get('quick_fix'):
        output.append("\n   ", style="dim")
        output.append("💡 QUICK FIX:", style="bold yellow")
        output.append("\n   ", style="dim")
        quick_fix_lines = parsed['quick_fix'].split('\n')
        for line in quick_fix_lines[:3]:  # Limit to 3 lines
            if line.strip():
                output.append(line.strip(), style="white")
                output.append("\n   ", style="dim")

    # Status
    if parsed.get('status'):
        output.append("\n   ", style="dim")
        output.append("📊 STATUS: ", style="bold yellow")
        status_text = parsed['status'].upper()
        if 'HEALTHY' in status_text:
            output.append(status_text, style="green bold")
        elif 'BROKEN' in status_text or 'INCOMPLETE' in status_text:
            output.append(status_text, style="red bold")
        elif 'NEEDS ATTENTION' in status_text:
            output.append(status_text, style="yellow bold")
        else:
            output.append(status_text, style="white bold")

    # Common errors
    if health.get('error_messages'):
        output.append("\n\n   ", style="dim")
        output.append("Common errors:", style="dim italic")
        for i, error in enumerate(health['error_messages'][:5], 1):
            output.append(f"\n   ", style="dim")
            # Truncate long errors
            error_display = error[:100] + "..." if len(error) > 100 else error
            output.append(f"- {error_display}", style="red")

    # AI Service/Model info
    output.append("\n\n   ", style="dim")
    output.append("🤖 AI Analysis: ", style="dim italic")
    output.append(f"{ai_service}", style="cyan")
    output.append(" / ", style="dim")
    output.append(f"{ai_model}", style="cyan italic")

    console.print(output)
    console.print()


def main():
    """Main entry point"""
    load_env()

    # Check for workflow ID argument
    workflow_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Display header
    console.print(Panel("Workflow Health Report", style="bold magenta"))
    
    # Show AI service configuration
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL')
    groq_api_key = os.getenv('GROQ_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    ai_status = []
    if ollama_model:
        ai_status.append(f"Ollama: {ollama_url} ({ollama_model})")
    else:
        ai_status.append("Ollama: Not configured (OLLAMA_MODEL missing)")
    
    if groq_api_key:
        groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        ai_status.append(f"Groq: {groq_model}")
    else:
        ai_status.append("Groq: Not configured")
    
    if gemini_api_key:
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        ai_status.append(f"Gemini: {gemini_model}")
    else:
        ai_status.append("Gemini: Not configured")
    
    console.print(f"[dim]AI Services: {' | '.join(ai_status)}[/dim]")
    console.print()

    # Fetch workflows
    workflows = fetch_workflows(workflow_id)

    if not workflows:
        console.print("[yellow]No workflows found[/yellow]")
        return

    # Track stats
    healthy = 0
    needs_attention = 0
    inactive = 0

    # Check each workflow
    for workflow in workflows:
        executions = fetch_executions(workflow['id'], limit=20)
        health = calculate_health(workflow, executions)
        structure = analyze_workflow_structure(workflow)
        ai_analysis = ask_ollama(workflow, health, structure)

        display_workflow_health(workflow, health, ai_analysis)

        # Extract response text from dict if needed
        ai_response_text = ai_analysis.get('response', '') if isinstance(ai_analysis, dict) else ai_analysis
        parsed = parse_ai_response(ai_response_text)
        ai_status = parsed.get('status', '').upper()

        # Check if fix is available and workflow needs attention
        implementation = parsed.get('implementation')
        if implementation and ai_status != 'HEALTHY':
            # Prompt for fix approval
            if prompt_for_fix_approval(workflow, implementation, parsed.get('quick_fix', '')):
                console.print("[cyan]Applying fix...[/cyan]")

                # Apply fix to workflow
                fix_result = apply_fix_to_workflow(workflow, implementation)

                if fix_result.get('success'):
                    updated_workflow = fix_result.get('workflow')

                    # Update workflow via API
                    api_result = update_workflow(workflow['id'], updated_workflow)

                    if api_result.get('success'):
                        console.print("[green]✓ Fix applied successfully![/green]")
                        console.print()
                    else:
                        console.print(f"[red]✗ Failed to update workflow via API: {api_result.get('error', 'Unknown error')}[/red]")
                        console.print()
                else:
                    console.print(f"[red]✗ Could not apply fix: {fix_result.get('error', 'Unknown error')}[/red]")
                    console.print()
            else:
                console.print("[yellow]Fix skipped[/yellow]")
                console.print()

        # Update stats based on AI analysis if available
        if not workflow.get('active'):
            inactive += 1
        elif 'HEALTHY' in ai_status or (not ai_status and health['success_rate'] >= 90):
            healthy += 1
        else:
            needs_attention += 1

    # Display summary
    console.print("─" * 40)
    console.print(f"Summary: {len(workflows)} workflow{'s' if len(workflows) != 1 else ''} checked")
    console.print(f"  • {healthy} healthy (🟢)")
    console.print(f"  • {needs_attention} needs attention (🟡)")
    console.print(f"  • {inactive} inactive (🔴)")


if __name__ == "__main__":
    main()
