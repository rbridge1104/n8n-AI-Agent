# N8N Workflow Health Checker

A Python CLI tool that monitors your n8n workflows and provides AI-powered improvement suggestions using Ollama.

## Features

- 📊 Check health status of all your n8n workflows
- ✅ Calculate success rates from recent executions
- ⏰ Track when workflows last ran
- 🤖 Get AI-powered improvement suggestions via Ollama or Groq
- 🔧 **Automatically implement fixes with user authorization**
- 🎨 Beautiful terminal output with Rich library
- 🔒 Secure credential management with `.env` files

## Quick Start

### Prerequisites

- Python 3.7+
- Access to an n8n instance with API enabled
- Ollama running locally or remotely

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd n8nPA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your credentials:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
N8N_URL=https://your-n8n-instance.com
N8N_API_KEY=your-api-key-here
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

### Getting Your n8n API Key

1. Open your n8n instance
2. Go to **Settings** → **API**
3. Click **Create API Key**
4. Copy the key to your `.env` file

## Usage

Check all workflows:
```bash
python workflow_health.py
```

Check a specific workflow:
```bash
python workflow_health.py <workflow-id>
```

## Output Example

```
╭────────────────────────────────────────╮
│ Workflow Health Report                 │
╰────────────────────────────────────────╯

🟢 Security RSS Aggregator
   Status: Active
   Success Rate: 98% (49/50 recent runs)
   Last Run: 12 minutes ago ✓

   💡 Quick Win: Add error handling to continue when one feed fails

🔴 Email Alerts Workflow
   Status: Inactive ⚠️
   Last Run: 3 days ago

   ⚠️ This workflow is turned off - activate it or delete it

────────────────────────────────────────
Summary: 2 workflows checked
  • 1 healthy (🟢)
  • 0 needs attention (🟡)
  • 1 inactive (🔴)
```

## How It Works

1. **Fetches workflows** from your n8n instance via API
2. **Retrieves execution history** (last 20 runs per workflow)
3. **Calculates health metrics**:
   - Active/Inactive status
   - Success rate percentage
   - Last successful run time
   - Error messages from failed executions
4. **Asks AI (Ollama/Groq)** for comprehensive analysis including:
   - Root cause analysis
   - Quick fix suggestions
   - Structured implementation plan
5. **Prompts for fix authorization** if a workflow needs attention
6. **Automatically applies fixes** when you approve them
7. **Displays results** in a beautiful, color-coded format

## Automatic Fix Implementation

When the AI detects issues with your workflows, it can now automatically fix them! The tool supports:

- **ADD_RETRY**: Add retry logic to failing nodes (configurable retries and delays)
- **UPDATE_NODE**: Modify node parameters (timeouts, credentials, settings)
- **FIX_SETTINGS**: Update workflow-level settings
- **ADD_ERROR_HANDLING**: Add error trigger nodes for better error management
- **ADD_NODE**: Insert new nodes into the workflow

### Fix Workflow

1. The AI analyzes your workflow and suggests a fix
2. You see exactly what changes will be made
3. You approve or decline the fix
4. If approved, the fix is applied to your n8n workflow automatically
5. You get immediate confirmation of success or error details

### Example

```
╭──────────────────────────────────────────╮
│ Fix Available for: Email Alerts Workflow│
╰──────────────────────────────────────────╯

Suggested Fix: Add retry logic to the Gmail node to handle temporary connection failures

Changes to be applied:
  • Add retry logic to node: Gmail
    - Max retries: 3
    - Wait between retries: 1000ms

Apply this fix? (yes/no): yes
Applying fix...
✓ Fix applied successfully!
```

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `N8N_URL` | Your n8n instance URL | `http://localhost:5678` |
| `N8N_API_KEY` | n8n API key | `your-api-key` |
| `OLLAMA_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model to use | `mistral:7b` |

## Status Icons

- 🟢 **Healthy**: Active workflow with ≥90% success rate
- 🟡 **Needs Attention**: Active workflow with <90% success rate
- 🔴 **Inactive**: Workflow is turned off

## Security

- Credentials are stored in `.env` (gitignored by default)
- Never commit your `.env` file
- Use `.env.example` as a template for others

## Dependencies

- `requests` - HTTP requests to n8n and Ollama APIs
- `python-dotenv` - Environment variable management
- `rich` - Beautiful terminal formatting

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

MIT

## Acknowledgments

Built with Claude Code 🤖
