import anthropic
import os
from dotenv import load_dotenv
import base64
import pypdf
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
def pdf_to_markdown(pdf_path):

    # Initialize PDF reader
    pdf_text = ""

    try:
        # Open and read PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + "\n\n"
        markdown_text = pdf_text
        paragraphs = markdown_text.split("\n\n")
        formatted_paragraphs = []

        for para in paragraphs:
            para = " ".join(para.split())
            if para.isupper() and len(para) < 100:
                formatted_paragraphs.append(f"## {para}")
            else:
                formatted_paragraphs.append(para)

        return "\n\n".join(formatted_paragraphs)

    except Exception as e:
        return f"Error converting PDF: {str(e)}"
    
def display_claude_message(message):
    """
    Displays Claude's message output in a beautifully formatted way using rich library.
    
    Args:
        message: The message object from Claude's response
    """
    console = Console()
    
    # Create main panel for message info
    message_info = Text()
    message_info.append(f"Message ID: ", style="bold cyan")
    message_info.append(f"{message.id}\n\n", style="cyan")
    message_info.append(f"Model: ", style="bold cyan")
    message_info.append(f"{message.model}\n", style="cyan")
    
    # Create table for content blocks
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Block Type", style="cyan")
    table.add_column("Content", style="white", overflow="fold")
    table.add_column("Citations", style="green")
    
    # Process each content block
    for block in message.content:
        # Get citations text if they exist
        citations = ""
        if block.citations:
            citations = "\n".join([
                f"Source: {cit.document_title}\n"
                f"Text: '{cit.cited_text}'"
                for cit in block.citations
            ])
        
        table.add_row(
            block.type,
            block.text,
            citations
        )
    
    # Create usage statistics panel
    usage_info = Text()
    usage_info.append(f"Input tokens: ", style="bold yellow")
    usage_info.append(f"{message.usage.input_tokens}\n", style="yellow")
    usage_info.append(f"Output tokens: ", style="bold yellow")
    usage_info.append(f"{message.usage.output_tokens}\n", style="yellow")
    
    # Display everything
    console.print("\n=== Claude Message Output ===\n", style="bold white on blue")
    console.print(Panel(message_info, title="Message Information", border_style="blue"))
    console.print("\n=== Content Blocks ===\n", style="bold white on blue")
    console.print(table)
    console.print("\n=== Usage Statistics ===\n", style="bold white on blue")
    console.print(Panel(usage_info, title="Usage Information", border_style="yellow"))


if __name__ == "__main__":

    markdown_content = pdf_to_markdown("data/relianceEarning.pdf")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": markdown_content,
                        },
                        "title": "Reliance Q1 Earning Transcript",
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "Which luxury store brands were mentioned in the meeting?"},
                ],
            }
        ],
    )

    display_claude_message(response)
