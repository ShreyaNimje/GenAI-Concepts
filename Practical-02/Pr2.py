import sys
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from rich.tree import Tree
from rich.prompt import Prompt
from rich.align import Align
from rich import box

console = Console()


# ============================================================
# RESOURCE SETUP
# ============================================================

def setup_resources():
    with console.status("[bold cyan]Loading NLP models & language packs...", spinner="dots"):
        packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]
        for p in packages:
            nltk.download(p, quiet=True)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp


# ============================================================
# UI COMPONENTS & DASHBOARDS
# ============================================================

def render_header():
    title = (
        "[bold white on blue]  🧠 NLP STUDIO  [/bold white on blue]\n"
        "[dim]Terminal Webapp • Python • spaCy • NLTK[/dim]"
    )
    console.print(Panel(Align.center(title), box=box.ROUNDED, border_style="blue"))


def render_metrics(text, sentences, tokens, filtered, ents):
    def metric_card(label, val, color):
        return Panel(
            Align.center(f"[bold {color}]{val}[/]\n[dim]{label}[/dim]"),
            border_style=color,
            box=box.ROUNDED,
            width=18
        )

    cols = Columns([
        metric_card("Characters", len(text), "cyan"),
        metric_card("Sentences", len(sentences), "green"),
        metric_card("Total Tokens", len(tokens), "yellow"),
        metric_card("Filtered (No Stop)", len(filtered), "magenta"),
        metric_card("Entities", len(ents), "red"),
    ], align="center")
    console.print(cols)


def render_dependency_tree(sent):
    """Builds a rich tree object from spaCy root token."""
    root_node = sent.root
    tree = Tree(f"[bold red]{root_node.text}[/] [dim]({root_node.pos_} • {root_node.dep_})[/dim]")

    def build_branch(token, parent_branch):
        for child in token.children:
            color = "cyan" if "subj" in child.dep_ else "yellow" if "obj" in child.dep_ else "green"
            node = parent_branch.add(
                f"[bold {color}]{child.text}[/] [dim]({child.pos_} • [italic]{child.dep_}[/italic])[/dim]"
            )
            build_branch(child, node)

    build_branch(root_node, tree)
    return tree


# ============================================================
# MAIN PIPELINE RUNNER
# ============================================================

def analyze_and_display(text, nlp):
    console.clear()
    render_header()

    with console.status("[bold magenta]Processing linguistic pipeline...", spinner="arc"):
        doc = nlp(text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered = [w for w in words if w.lower() not in stop_words and w.isalnum()]

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(words)

    # 1. Text Summary Card
    console.print(Panel(f"[italic]{text}[/italic]", title="[bold]📝 Active Document[/bold]", border_style="dim", box=box.ROUNDED))
    render_metrics(text, sentences, words, filtered, doc.ents)
    console.print()

    # 2. Tabs / View Selector Loop
    while True:
        console.print("[bold cyan]Select a View / Module:[/bold cyan]")
        console.print(" [1] 📑 Tokenization & Segmentation\n"
                      " [2] 🌱 Stemming vs Lemmatization\n"
                      " [3] 🏷️  Part-of-Speech (POS) Tags\n"
                      " [4] 🔎 Named Entity Recognition (NER)\n"
                      " [5] 🌳 Visual Dependency Trees\n"
                      " [0] 🔄 Analyze New Text / Exit")
        
        choice = Prompt.ask("\n[bold yellow]Navigate to[/bold yellow]", choices=["1", "2", "3", "4", "5", "0"], default="1")

        if choice == "0":
            break

        console.print()

        # VIEW 1: Basic Processing
        if choice == "1":
            t = Table(title="Sentence & Word Segmentation", box=box.SIMPLE_HEAVY, border_style="cyan")
            t.add_column("Level", style="bold cyan", width=15)
            t.add_column("Output Preview", style="white")
            
            for i, s in enumerate(sentences, 1):
                t.add_row(f"Sentence {i}", s)
            t.add_section()
            t.add_row("Total Tokens", f"[yellow]{', '.join(words[:25])}{'...' if len(words) > 25 else ''}[/yellow]")
            t.add_row("Cleaned Tokens", f"[green]{', '.join(filtered[:25])}{'...' if len(filtered) > 25 else ''}[/green]")
            console.print(t)

        # VIEW 2: Stemming vs Lemmatization
        elif choice == "2":
            t = Table(title="Morphological Normalization Comparison", box=box.SIMPLE_HEAVY, border_style="magenta")
            t.add_column("Original Word", style="bold white")
            t.add_column("Porter Stemmer (Rule-based)", style="yellow")
            t.add_column("WordNet Lemmatizer (Lexical)", style="green")

            for word in filtered:
                t.add_row(word, stemmer.stem(word), lemmatizer.lemmatize(word))
            console.print(t)

        # VIEW 3: POS Tags
        elif choice == "3":
            t = Table(title="Part-of-Speech Tagging", box=box.SIMPLE_HEAVY, border_style="yellow")
            t.add_column("Token", style="bold white")
            t.add_column("Penn Tag", style="bold cyan")
            t.add_column("Grammatical Category", style="dim white")

            for word, tag in pos_tags:
                t.add_row(word, tag, spacy.explain(tag) or "Punctuation/Symbol")
            console.print(t)

        # VIEW 4: Named Entity Recognition
        elif choice == "4":
            if not doc.ents:
                console.print(Panel("[yellow]No Named Entities Detected.[/yellow]", border_style="yellow"))
            else:
                t = Table(title="Entity Identification (NER)", box=box.SIMPLE_HEAVY, border_style="red")
                t.add_column("Entity Text", style="bold white")
                t.add_column("Label Type", style="bold red")
                t.add_column("Description", style="dim white")

                for ent in doc.ents:
                    t.add_row(ent.text, ent.label_, spacy.explain(ent.label_))
                console.print(t)

        # VIEW 5: Visual Tree Diagram
        elif choice == "5":
            for i, sent in enumerate(doc.sents, 1):
                tree_panel = Panel(
                    render_dependency_tree(sent),
                    title=f"[bold green]Sentence {i} Syntactic Tree[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                )
                console.print(tree_panel)

        console.print("\n" + "─" * 70 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    nlp = setup_resources()
    default_text = (
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in California. "
        "The company is currently developing advanced artificial intelligence tools."
    )

    while True:
        console.clear()
        render_header()
        console.print("[bold cyan]Enter your paragraph below[/bold cyan] (Press [bold yellow]Enter[/bold yellow] to use sample text):\n")
        user_input = console.input("[bold green]>> [/bold green]").strip()
        
        selected_text = user_input if user_input else default_text
        analyze_and_display(selected_text, nlp)
        
        again = Prompt.ask("\n[bold cyan]Would you like to analyze another text?[/bold cyan]", choices=["y", "n"], default="y")
        if again == "n":
            console.print("\n[bold blue]Exiting NLP Studio. Goodbye! 🧠[/bold blue]")
            break


if __name__ == "__main__":
    main()