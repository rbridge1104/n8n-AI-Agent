#!/usr/bin/env python3
"""
N8N Workflow Health Checker
Checks workflow health and suggests improvements using Ollama AI
"""

import sys
import os
import json
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


def fetch_executions(workflow_id, limit=10):
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


def calculate_health(workflow, executions):
    """Calculate workflow health metrics"""
    total = len(executions)
    if total == 0:
        return {
            'success_rate': 0,
            'success_count': 0,
            'total_count': 0,
            'last_run': None,
            'last_success': None
        }

    successful = sum(1 for e in executions if e.get('finished') and not e.get('stoppedAt'))

    # Find last run and last successful run
    last_run = None
    last_success = None

    for execution in executions:
        stopped_at = execution.get('stoppedAt')
        if stopped_at:
            if not last_run:
                last_run = stopped_at
            if execution.get('finished') and not execution.get('stoppedAt'):
                if not last_success:
                    last_success = stopped_at

    return {
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'success_count': successful,
        'total_count': total,
        'last_run': last_run,
        'last_success': last_success
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


def ask_ollama(workflow, health):
    """Ask Ollama (or Groq fallback) for improvement suggestion"""
    active_status = "active" if workflow.get('active') else "inactive"

    prompt = f"""This is an n8n automation workflow called '{workflow['name']}'.
Status: {active_status}
Success rate: {health['success_rate']:.0f}% over last {health['total_count']} runs
Last successful run: {time_ago(health['last_success'])}

Suggest ONE quick, specific improvement to make it more reliable.
Be practical and concise (2-3 sentences max).

Workflow JSON: {json.dumps(workflow.get('nodes', [])[:3])}"""

    # Try Ollama first
    ollama_url = os.getenv('OLLAMA_URL')
    if ollama_url:
        try:
            base_url = ollama_url.rstrip('/')
            url = f"{base_url}/api/generate"
            payload = {
                "model": os.getenv('OLLAMA_MODEL'),
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('response', 'Unable to generate suggestion')
        except requests.RequestException:
            pass  # Fall through to Groq

    # Fallback to Groq if Ollama fails or not configured
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
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException:
            pass  # Both failed

    return "Unable to connect to AI service (tried Ollama and Groq)"


def get_status_icon(workflow, health):
    """Get status icon based on workflow health"""
    if not workflow.get('active'):
        return "🔴"
    elif health['success_rate'] >= 90:
        return "🟢"
    else:
        return "🟡"


def display_workflow_health(workflow, health, suggestion):
    """Display health information for a single workflow"""
    icon = get_status_icon(workflow, health)

    output = Text()
    output.append(f"{icon} ", style="bold")
    output.append(workflow['name'], style="bold cyan")
    output.append("\n")

    # Status
    status = "Active" if workflow.get('active') else "Inactive"
    status_color = "green" if workflow.get('active') else "red"
    output.append(f"   Status: ", style="dim")
    output.append(status, style=status_color)

    if not workflow.get('active'):
        output.append(" ⚠️", style="yellow")
    output.append("\n")

    # Success rate (only if active)
    if workflow.get('active') and health['total_count'] > 0:
        rate_color = "green" if health['success_rate'] >= 90 else "yellow" if health['success_rate'] >= 70 else "red"
        output.append(f"   Success Rate: ", style="dim")
        output.append(f"{health['success_rate']:.0f}% ({health['success_count']}/{health['total_count']} recent runs)",
                     style=rate_color)
        output.append("\n")

    # Last run
    output.append(f"   Last Run: ", style="dim")
    last_run_time = time_ago(health['last_run'])
    output.append(last_run_time)
    if health['last_run']:
        output.append(" ✓", style="green")
    output.append("\n")

    # Suggestion
    output.append("\n   ", style="dim")
    if not workflow.get('active'):
        output.append("⚠️  This workflow is turned off - activate it or delete it", style="yellow italic")
    else:
        output.append("💡 Quick Win: ", style="bold yellow")
        output.append(suggestion.strip(), style="italic")

    console.print(output)
    console.print()


def main():
    """Main entry point"""
    load_env()

    # Check for workflow ID argument
    workflow_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Display header
    console.print(Panel("Workflow Health Report", style="bold magenta"))
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
        executions = fetch_executions(workflow['id'])
        health = calculate_health(workflow, executions)
        suggestion = ask_ollama(workflow, health)

        display_workflow_health(workflow, health, suggestion)

        # Update stats
        if not workflow.get('active'):
            inactive += 1
        elif health['success_rate'] >= 90:
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
