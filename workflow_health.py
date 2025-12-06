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

    # At least one AI service required (Ollama or Groq)
    has_ollama = os.getenv('OLLAMA_URL') and os.getenv('OLLAMA_MODEL')
    has_groq = os.getenv('GROQ_API_KEY')

    if not has_ollama and not has_groq:
        console.print("[red]Error: At least one AI service must be configured:[/red]")
        console.print("[yellow]  - Ollama (OLLAMA_URL + OLLAMA_MODEL), or[/yellow]")
        console.print("[yellow]  - Groq (GROQ_API_KEY)[/yellow]")
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

    # n8n API only accepts specific fields for workflow updates
    # Only include fields that are allowed by the API
    allowed_fields = ['name', 'nodes', 'connections', 'active', 'settings', 'staticData', 'tags', 'pinData']
    clean_data = {}
    
    # Name is required - ensure it's included
    if 'name' not in workflow_data:
        # Try to get name from original workflow if available
        console.print("[yellow]Warning: 'name' field missing, attempting to fetch from workflow...[/yellow]")
        try:
            original = fetch_workflows(workflow_id)
            if original and len(original) > 0:
                workflow_data['name'] = original[0].get('name', 'Untitled Workflow')
        except:
            workflow_data['name'] = 'Untitled Workflow'
    
    for field in allowed_fields:
        if field in workflow_data:
            clean_data[field] = workflow_data[field]
    
    # Ensure name is always present (required by API)
    if 'name' not in clean_data:
        return {'success': False, 'error': 'Workflow name is required but was not provided'}
    
    # Clean nodes - remove read-only fields that cause API errors
    if 'nodes' in clean_data:
        for node in clean_data['nodes']:
            # Remove read-only fields
            for field in ['id', 'createdAt', 'updatedAt', 'versionId', 'meta', 'webhookId']:
                node.pop(field, None)
            # Clean credentials to only include id and name
            if 'credentials' in node and isinstance(node['credentials'], dict):
                cleaned_creds = {}
                for cred_type, cred_data in node['credentials'].items():
                    if isinstance(cred_data, dict):
                        cleaned_creds[cred_type] = {
                            'id': cred_data.get('id'),
                            'name': cred_data.get('name')
                        }
                    else:
                        cleaned_creds[cred_type] = cred_data
                node['credentials'] = cleaned_creds

    # Debug: Show what we're sending (temporarily)
    console.print("[dim]Debug: Sending fields to n8n API:[/dim]", style="dim")
    console.print(f"[dim]  - Fields: {list(clean_data.keys())}[/dim]", style="dim")
    if 'nodes' in clean_data and len(clean_data['nodes']) > 0:
        sample_node = clean_data['nodes'][0]
        console.print(f"[dim]  - Sample node fields: {list(sample_node.keys())}[/dim]", style="dim")

    # Save to file for inspection
    try:
        with open('debug_workflow_update.json', 'w') as f:
            json.dump(clean_data, f, indent=2)
        console.print("[dim]  - Full data saved to debug_workflow_update.json[/dim]", style="dim")
    except:
        pass

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
                if hasattr(e.response, 'text'):
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
            'never_succeeded': True
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

    return {
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'success_count': successful,
        'failed_count': failed,
        'total_count': total,
        'last_run': last_run,
        'last_success': last_success,
        'never_succeeded': successful == 0 and total > 0
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


def ask_ollama(workflow, health, structure):
    """Ask Ollama (or Groq/Gemini fallback) for comprehensive workflow analysis with implementation plan"""
    active_status = "active" if workflow.get('active') else "inactive"
    
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
- Never succeeded: {'Yes' if health.get('never_succeeded') else 'No'}

WORKFLOW STRUCTURE:
- Total nodes: {structure['node_count']}
- Node types: {node_types_list}
- Has error handling: {'Yes' if structure['has_error_handling'] else 'No'}
- Has retry logic: {'Yes' if structure['has_retry_logic'] else 'No'}

WORKFLOW JSON:
{json.dumps(workflow, indent=2)}

Based on this analysis, provide:

1. ROOT CAUSE: What's actually wrong with this workflow? Be specific.
   - If it's never run successfully, is it incomplete or misconfigured?
   - If it's failing consistently, what's the likely issue?
   - If success rate is low, what pattern do you see?

2. QUICK FIX: ONE specific action to improve reliability (5-30 min to implement)

3. IMPLEMENTATION: Provide concrete, implementable changes. Choose ONE approach:
   - UPDATE_NODE: Update a node's parameters (e.g., timeout, credentials, settings).
   - ADD_NODE: Add a new node to the workflow (e.g., IF node for error handling).
   - FIX_SETTINGS: Update workflow-level settings.
   - ADD_ERROR_HANDLING: Add an error workflow or error trigger node.
   - REMOVE_NODE: Remove a redundant or problematic node.
   - UPDATE_CONNECTIONS: Fix or update node connections.

   NOTE: Do NOT suggest retry settings (retryOnFail, maxTries) - these cannot be set via API.
   Instead suggest timeout increases, error handling nodes, or other workflow changes.

   Format as JSON:
   {{"action": "UPDATE_NODE|ADD_NODE|FIX_SETTINGS|ADD_ERROR_HANDLING|REMOVE_NODE|UPDATE_CONNECTIONS", "details": {{"node_name": "...", "changes": {{...}}}}}}

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
        except requests.RequestException:
            pass  # Fall through to Groq/Gemini

    # Fallback to Groq if Ollama fails
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
                "max_tokens": 1500
            }
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']
            return {
                'response': response_text,
                'service': 'Groq',
                'model': groq_model
            }
        except requests.RequestException:
            pass

    # Fallback to Gemini
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    if gemini_api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0].get('content', {})
                parts = content.get('parts', [])
                if parts and 'text' in parts[0]:
                    return {
                        'response': parts[0]['text'],
                        'service': 'Gemini',
                        'model': gemini_model
                    }
        except requests.RequestException:
            pass

    return {
        'response': "Unable to connect to AI service (tried Ollama, Groq, and Gemini)",
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
            elif 'IMPLEMENTATION:' in response_text:
                # Try to extract JSON after IMPLEMENTATION:
                impl_start = response_text.find('IMPLEMENTATION:') + len('IMPLEMENTATION:')
                impl_text = response_text[impl_start:].strip()
                # Look for JSON object
                json_start = impl_text.find('{')
                if json_start >= 0:
                    json_end = impl_text.rfind('}') + 1
                    if json_end > json_start:
                        json_str = impl_text[json_start:json_end]
                        result['implementation'] = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Extract STATUS
    if 'STATUS:' in response_text:
        parts = response_text.split('STATUS:')
        if len(parts) > 1:
            result['status'] = parts[1].strip().split('\n')[0].strip()
    
    # If parsing failed, return raw response as root cause
    if not any([result['root_cause'], result['quick_fix'], result['status']]):
        result['root_cause'] = response_text
    
    return result


def generate_workflow_modifications(workflow, implementation, ai_service='Ollama'):
    """Use AI to generate actual workflow JSON modifications based on implementation plan"""
    if not implementation or 'action' not in implementation:
        return {'success': False, 'error': 'No implementation provided'}
    
    action = implementation.get('action', '')
    details = implementation.get('details', {})
    
    prompt = f"""You are an n8n workflow expert. Generate the exact JSON modifications needed to implement this change:

Action: {action}
Details: {json.dumps(details, indent=2)}

Current Workflow JSON:
{json.dumps(workflow, indent=2)}

Generate the MODIFIED workflow JSON with ONLY the changes needed. Return ONLY valid JSON that can be used to update the workflow via n8n API.

Important:
- Only include fields allowed by n8n API: name, nodes, connections, active, settings, staticData, tags, pinData
- Do NOT include read-only fields: id, createdAt, updatedAt, versionId, meta
- Preserve all existing nodes and connections unless explicitly changing them
- Make minimal changes - only what's needed for the implementation

Return ONLY the JSON object, no explanation."""

    # Use the same AI service that was used for analysis
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL')
    
    if ai_service == 'Ollama' and ollama_model:
        try:
            base_url = ollama_url.rstrip('/')
            url = f"{base_url}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            response_text = response.json().get('response', '')
        except requests.RequestException:
            response_text = None
    else:
        response_text = None
    
    # Fallback to Groq
    if not response_text:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,  # Lower temperature for more precise JSON
                    "max_tokens": 4000
                }
                response = requests.post(url, json=payload, headers=headers, timeout=90)
                response.raise_for_status()
                response_text = response.json()['choices'][0]['message']['content']
            except requests.RequestException:
                return {'success': False, 'error': 'Failed to connect to AI service'}
    
    # Try to extract JSON from response
    try:
        # Try to find JSON in the response
        json_start = response_text.find('{')
        if json_start >= 0:
            json_end = response_text.rfind('}') + 1
            if json_end > json_start:
                json_str = response_text[json_start:json_end]
                modified_workflow = json.loads(json_str)
                
                # Clean the workflow - ensure it has required fields and only allowed fields
                allowed_fields = ['name', 'nodes', 'connections', 'active', 'settings', 'staticData', 'tags', 'pinData']
                clean_workflow = {}
                
                # Always include name (required by API) - use original if AI didn't include it
                if 'name' in modified_workflow:
                    clean_workflow['name'] = modified_workflow['name']
                elif workflow.get('name'):
                    clean_workflow['name'] = workflow['name']
                else:
                    return {'success': False, 'error': 'Workflow name is required but missing'}
                
                # Include other allowed fields if they exist in AI response
                for field in allowed_fields:
                    if field != 'name' and field in modified_workflow:
                        clean_workflow[field] = modified_workflow[field]
                
                # Clean nodes - remove read-only fields
                if 'nodes' in clean_workflow:
                    for node in clean_workflow['nodes']:
                        for field in ['id', 'createdAt', 'updatedAt', 'versionId', 'meta', 'webhookId']:
                            node.pop(field, None)
                        # Clean credentials
                        if 'credentials' in node and isinstance(node['credentials'], dict):
                            cleaned_creds = {}
                            for cred_type, cred_data in node['credentials'].items():
                                if isinstance(cred_data, dict):
                                    cleaned_creds[cred_type] = {
                                        'id': cred_data.get('id'),
                                        'name': cred_data.get('name')
                                    }
                                else:
                                    cleaned_creds[cred_type] = cred_data
                            node['credentials'] = cleaned_creds
                
                return {'success': True, 'workflow': clean_workflow}
    except (json.JSONDecodeError, ValueError) as e:
        return {'success': False, 'error': f'Failed to parse AI response as JSON: {str(e)}'}
    
    return {'success': False, 'error': 'No valid JSON found in AI response'}


def apply_workflow_fix(workflow, implementation):
    """Apply AI-suggested fix directly to workflow JSON (simpler approach)"""
    if not implementation or 'action' not in implementation:
        return {'success': False, 'error': 'No implementation provided'}

    action = implementation.get('action', '')
    details = implementation.get('details', {})
    
    # Make a deep copy to avoid modifying the original
    try:
        updated_workflow = copy.deepcopy(workflow)
        nodes = updated_workflow.get('nodes', [])
        connections = updated_workflow.get('connections', {})
    except Exception as e:
        return {'success': False, 'error': f'Failed to copy workflow: {str(e)}'}

    if action == 'ADD_RETRY':
        # Note: n8n API doesn't support setting retry configuration via workflow JSON
        # Retry settings (retryOnFail, maxTries, waitBetweenTries) are execution settings
        # that must be configured in the n8n UI, not via the API
        return {
            'success': False,
            'error': 'ADD_RETRY is not supported via n8n API. Retry settings must be configured manually in the n8n workflow editor. You can set these in the node settings under "Settings" > "Retry On Fail".'
        }
    
    elif action == 'UPDATE_NODE':
        node_name = details.get('node_name', '')
        if not node_name:
            return {'success': False, 'error': 'Node name not specified'}
        
        changes = details.get('changes', {})
        node_found = False
        
        for node in nodes:
            if node.get('name') == node_name:
                node_found = True
                if 'parameters' not in node:
                    node['parameters'] = {}
                for key, value in changes.items():
                    node['parameters'][key] = value
                break
        
        if not node_found:
            return {'success': False, 'error': f'Node "{node_name}" not found in workflow'}
    
    elif action == 'REMOVE_NODE':
        node_name = details.get('node_name', '')
        if not node_name:
            return {'success': False, 'error': 'Node name not specified'}
        
        # Remove node
        nodes[:] = [n for n in nodes if n.get('name') != node_name]
        
        # Remove connections involving this node
        connections_to_remove = []
        for source_node, outputs in connections.items():
            if source_node == node_name:
                connections_to_remove.append(source_node)
            else:
                for output_type, output_connections in outputs.items():
                    for connection_list in output_connections:
                        connection_list[:] = [c for c in connection_list if c.get('node') != node_name]
        
        for node_to_remove in connections_to_remove:
            del connections[node_to_remove]
    
    elif action == 'UPDATE_CONNECTIONS':
        # Update connections based on details
        new_connections = details.get('connections', {})
        if new_connections:
            updated_workflow['connections'] = new_connections
    
    elif action == 'FIX_SETTINGS':
        settings = details.get('changes', {})
        if not settings:
            return {'success': False, 'error': 'No settings changes specified'}
        
        if 'settings' not in updated_workflow:
            updated_workflow['settings'] = {}
        
        for key, value in settings.items():
            updated_workflow['settings'][key] = value
    
    # Clean the workflow data for API - only include allowed fields
    allowed_fields = ['name', 'nodes', 'connections', 'active', 'settings', 'staticData', 'tags', 'pinData']
    clean_workflow = {}
    
    # Always include name (required by API)
    if 'name' in updated_workflow:
        clean_workflow['name'] = updated_workflow['name']
    elif workflow.get('name'):
        clean_workflow['name'] = workflow['name']
    else:
        return {'success': False, 'error': 'Workflow name is required but missing'}
    
    # Include other allowed fields if they exist
    for field in allowed_fields:
        if field != 'name' and field in updated_workflow:
            clean_workflow[field] = updated_workflow[field]
    
    # Clean nodes - remove read-only fields
    if 'nodes' in clean_workflow:
        for node in clean_workflow['nodes']:
            # Remove read-only fields that cause API errors
            for field in ['id', 'createdAt', 'updatedAt', 'versionId', 'meta', 'webhookId']:
                node.pop(field, None)
            # Also clean credentials if present (they're stored separately)
            if 'credentials' in node:
                # Keep credentials but ensure they're in the right format
                creds = node['credentials']
                if isinstance(creds, dict):
                    # Keep only id and name for credentials
                    cleaned_creds = {}
                    for cred_type, cred_data in creds.items():
                        if isinstance(cred_data, dict):
                            cleaned_creds[cred_type] = {
                                'id': cred_data.get('id'),
                                'name': cred_data.get('name')
                            }
                        else:
                            cleaned_creds[cred_type] = cred_data
                    node['credentials'] = cleaned_creds
    
    return {'success': True, 'workflow': clean_workflow}


def prompt_for_approval(workflow, quick_fix, implementation):
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
    elif action == 'REMOVE_NODE':
        console.print(f"  • Remove node: [yellow]{details.get('node_name', 'Unknown')}[/yellow]")
    elif action == 'UPDATE_CONNECTIONS':
        console.print("  • Update workflow connections")
    elif action == 'FIX_SETTINGS':
        console.print("  • Update workflow settings")
        changes = details.get('changes', {})
        if changes:
            for key, value in changes.items():
                console.print(f"    - {key}: {value}")
    else:
        console.print(f"  • Action: {action}")
    
    console.print()
    
    # Prompt for approval
    try:
        response = input("[bold green]Apply this fix? (yes/no):[/bold green] ").strip().lower()
        return response in ['yes', 'y']
    except (KeyboardInterrupt, EOFError):
        return False


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
        
        if health.get('never_succeeded'):
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
        root_cause_lines = parsed['root_cause'].split('\n')
        for line in root_cause_lines[:5]:
            if line.strip():
                output.append(line.strip(), style="white")
                output.append("\n   ", style="dim")

    # Quick Fix
    if parsed.get('quick_fix'):
        output.append("\n   ", style="dim")
        output.append("💡 QUICK FIX:", style="bold yellow")
        output.append("\n   ", style="dim")
        quick_fix_lines = parsed['quick_fix'].split('\n')
        for line in quick_fix_lines[:3]:
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

    # AI Service/Model info
    output.append("\n\n   ", style="dim")
    output.append("🤖 AI Analysis: ", style="dim italic")
    output.append(f"{ai_service}", style="cyan")
    output.append(" / ", style="dim")
    output.append(f"{ai_model}", style="cyan italic")

    console.print(output)
    console.print()
    
    return parsed


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

        parsed = display_workflow_health(workflow, health, ai_analysis)

        # Check if fix is available and workflow needs attention
        implementation = parsed.get('implementation')
        ai_status_text = parsed.get('status', '').upper()
        
        if implementation and ai_status_text not in ['HEALTHY', '']:
            # Prompt for fix approval
            if prompt_for_approval(workflow, parsed.get('quick_fix', ''), implementation):
                console.print("[cyan]Applying fix...[/cyan]")
                
                # Apply fix to workflow
                fix_result = apply_workflow_fix(workflow, implementation)
                
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
                        # Try using AI to generate full workflow JSON as fallback
                        console.print("[yellow]Attempting AI-generated workflow update...[/yellow]")
                        ai_service = ai_analysis.get('service', 'Ollama') if isinstance(ai_analysis, dict) else 'Ollama'
                        ai_result = generate_workflow_modifications(workflow, implementation, ai_service)
                        if ai_result.get('success'):
                            api_result2 = update_workflow(workflow['id'], ai_result['workflow'])
                            if api_result2.get('success'):
                                console.print("[green]✓ Fix applied using AI-generated workflow![/green]")
                            else:
                                console.print(f"[red]✗ AI-generated update also failed: {api_result2.get('error')}[/red]")
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
        elif 'HEALTHY' in ai_status_text or (not ai_status_text and health['success_rate'] >= 90):
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