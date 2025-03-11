import os
import click
import cmd
import index
import search

@click.group()
def cli():
    """Claude.ai Conversation Search - Search your Claude conversation history semantically."""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--batch-size', default=1000, help='Batch size for indexing operations')
@click.option('--checkpoint-interval', default=5000, help='Number of messages to process before saving a checkpoint')
def index_conversations(file_path: str, batch_size: int, checkpoint_interval: int):
    """Index conversations from a Claude.ai export file."""
    click.echo(f"Indexing conversations from {file_path}")
    index.index_conversations(file_path, batch_size, checkpoint_interval)
    click.echo("Indexing complete!")

@cli.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
def search_conversations(query: str, top_k: int):
    """Search your indexed Claude.ai conversations."""
    click.echo(f"Searching for: {query}")
    
    # Check if index exists
    if not os.path.exists(index.CHROMA_DIR):
        click.echo("No index found. Please run 'index-conversations' first.")
        return
    
    results = search.search_and_display(
        query=query,
        top_k=top_k
    )
    
    if not results:
        click.echo("No results found.")

@cli.command()
def stats():
    """Show statistics about the indexed conversations."""
    try:
        client = search.setup_chroma_client()
        collection = search.get_collection(client)
        count = collection.count()
        click.echo(f"Total indexed messages: {count}")
    except Exception as e:
        click.echo(f"Error getting stats: {e}")

class SearchShell(cmd.Cmd):
    """Interactive shell for Claude conversation search."""
    intro = "Welcome to Claude Search Shell. Type 'help' for commands, 'exit' to quit.\n"
    prompt = "claude-search> "
    
    def __init__(self):
        super().__init__()
        # Initialize model on startup
        search.get_model()
        print("Model loaded and ready. Searches will now be much faster!")
        
        # Check if index exists
        if not os.path.exists(index.CHROMA_DIR):
            print("No index found. Please run 'index-conversations' first.")
    
    def do_search(self, arg):
        """Search conversations: search [query] --top-k=[number]"""
        args = arg.split()
        if not args:
            print("Please provide a search query")
            return
            
        # Parse for --top-k option
        top_k = DEFAULT_TOP_K
        query_parts = []
        
        for part in args:
            if part.startswith("--top-k="):
                try:
                    top_k = int(part.split("=")[1])
                except (ValueError, IndexError):
                    print(f"Invalid top-k value: {part}")
                    return
            else:
                query_parts.append(part)
                
        query = " ".join(query_parts)
        if not query:
            print("Please provide a search query")
            return
            
        print(f"Searching for: {query}")
        results = search.search_and_display(query=query, top_k=top_k)
        
        if not results:
            print("No results found.")
    
    def do_stats(self, arg):
        """Show statistics about the indexed conversations."""
        try:
            client = search.setup_chroma_client()
            collection = search.get_collection(client)
            count = collection.count()
            print(f"Total indexed messages: {count}")
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def do_exit(self, arg):
        """Exit the program."""
        print("Goodbye!")
        return True
        
    def do_quit(self, arg):
        """Exit the program."""
        return self.do_exit(arg)
        
    def default(self, line):
        """Default behavior: treat as search query."""
        if line.strip():
            self.do_search(line)

# Default top-k value for the shell
DEFAULT_TOP_K = 5

@cli.command()
def shell():
    """Start an interactive search shell (keeps model loaded between searches)."""
    SearchShell().cmdloop()

if __name__ == '__main__':
    cli()