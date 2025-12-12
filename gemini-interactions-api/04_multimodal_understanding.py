"""
Multimodal Understanding Example
Demonstrates image, audio, video, and document analysis using the Interactions API.
"""

from google import genai
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class MultimodalAnalyzer:
    """Handles multimodal content analysis."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def analyze_image_from_file(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image from a local file.
        
        Args:
            image_path: Path to the image file
            prompt: Question or instruction about the image
            
        Returns:
            Model's analysis
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine MIME type
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(extension, 'image/png')
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": base64_image, "mime_type": mime_type}
            ]
        )
        
        return interaction.outputs[-1].text
    
    def analyze_image_from_url(self, image_url: str, prompt: str) -> str:
        """
        Analyze an image from a URL.
        
        Args:
            image_url: URL of the image
            prompt: Question or instruction about the image
            
        Returns:
            Model's analysis
        """
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "uri": image_url}
            ]
        )
        
        return interaction.outputs[-1].text
    
    def analyze_audio(self, audio_path: str, prompt: str) -> str:
        """
        Analyze an audio file.
        
        Args:
            audio_path: Path to the audio file
            prompt: Question or instruction about the audio
            
        Returns:
            Model's analysis
        """
        # Read and encode audio
        with open(audio_path, "rb") as f:
            base64_audio = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine MIME type
        extension = Path(audio_path).suffix.lower()
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.m4a': 'audio/m4a',
            '.ogg': 'audio/ogg'
        }
        mime_type = mime_types.get(extension, 'audio/wav')
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "audio", "data": base64_audio, "mime_type": mime_type}
            ]
        )
        
        return interaction.outputs[-1].text
    
    def analyze_video(self, video_path: str, prompt: str) -> str:
        """
        Analyze a video file.
        
        Args:
            video_path: Path to the video file
            prompt: Question or instruction about the video
            
        Returns:
            Model's analysis
        """
        # Read and encode video
        with open(video_path, "rb") as f:
            base64_video = base64.b64encode(f.read()).decode('utf-8')
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "video", "data": base64_video, "mime_type": "video/mp4"}
            ]
        )
        
        return interaction.outputs[-1].text
    
    def analyze_document(self, pdf_path: str, prompt: str) -> str:
        """
        Analyze a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            prompt: Question or instruction about the document
            
        Returns:
            Model's analysis
        """
        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "document", "data": base64_pdf, "mime_type": "application/pdf"}
            ]
        )
        
        return interaction.outputs[-1].text
    
    def compare_images(self, image1_path: str, image2_path: str, prompt: str) -> str:
        """
        Compare two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            prompt: Comparison question
            
        Returns:
            Model's comparison
        """
        # Encode both images
        with open(image1_path, "rb") as f:
            base64_image1 = base64.b64encode(f.read()).decode('utf-8')
        
        with open(image2_path, "rb") as f:
            base64_image2 = base64.b64encode(f.read()).decode('utf-8')
        
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": base64_image1, "mime_type": "image/png"},
                {"type": "image", "data": base64_image2, "mime_type": "image/png"}
            ]
        )
        
        return interaction.outputs[-1].text


class FilesAPIAnalyzer:
    """Uses Gemini Files API for large file handling."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def analyze_large_file(self, file_path: str, prompt: str) -> str:
        """
        Analyze a large file using the Files API.
        
        Args:
            file_path: Path to the file
            prompt: Question or instruction
            
        Returns:
            Model's analysis
        """
        # Upload file to Gemini Files API
        print(f"Uploading {file_path}...")
        uploaded_file = self.client.files.upload(file=file_path)
        
        # Wait for processing
        import time
        while self.client.files.get(name=uploaded_file.name).state != "ACTIVE":
            print("Processing file...")
            time.sleep(2)
        
        print("File ready for analysis.")
        
        # Determine content type
        extension = Path(file_path).suffix.lower()
        if extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            content_type = "image"
        elif extension in ['.mp4', '.mov', '.avi']:
            content_type = "video"
        elif extension in ['.wav', '.mp3', '.m4a']:
            content_type = "audio"
        elif extension == '.pdf':
            content_type = "document"
        else:
            content_type = "image"  # Default
        
        # Analyze using Files API URI
        interaction = self.client.interactions.create(
            model="gemini-2.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": content_type, "uri": uploaded_file.uri}
            ]
        )
        
        return interaction.outputs[-1].text


def demo_image_analysis():
    """Demonstrate image analysis capabilities."""
    print("=" * 60)
    print("Image Analysis Examples")
    print("=" * 60)
    
    analyzer = MultimodalAnalyzer()
    
    # Example 1: Analyze from URL
    print("\n1. Analyzing Image from URL:")
    print("-" * 60)
    
    # Using a sample image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    try:
        response = analyzer.analyze_image_from_url(
            image_url,
            "Describe what you see in this image in detail."
        )
        print(f"Analysis: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("(Note: Ensure you have internet connection for URL-based images)")
    
    # Example 2: Analyze local file (if exists)
    print("\n2. Analyzing Local Image:")
    print("-" * 60)
    print("(Skipped - requires local image file)")
    print("To use: analyzer.analyze_image_from_file('path/to/image.png', 'Describe this image')")


def demo_audio_analysis():
    """Demonstrate audio analysis capabilities."""
    print("\n" + "=" * 60)
    print("Audio Analysis Examples")
    print("=" * 60)
    
    print("\nAudio Transcription:")
    print("-" * 60)
    print("(Skipped - requires local audio file)")
    print("To use: analyzer.analyze_audio('path/to/audio.wav', 'Transcribe this audio')")
    
    print("\nExample use cases:")
    print("- Transcribe speech to text")
    print("- Identify music genre")
    print("- Detect emotions in voice")
    print("- Extract key information from recordings")


def demo_video_analysis():
    """Demonstrate video analysis capabilities."""
    print("\n" + "=" * 60)
    print("Video Analysis Examples")
    print("=" * 60)
    
    print("\nVideo Understanding:")
    print("-" * 60)
    print("(Skipped - requires local video file)")
    print("To use: analyzer.analyze_video('path/to/video.mp4', 'Summarize this video')")
    
    print("\nExample use cases:")
    print("- Generate video summaries")
    print("- Create timestamped descriptions")
    print("- Identify objects and actions")
    print("- Extract text from video")


def demo_document_analysis():
    """Demonstrate document analysis capabilities."""
    print("\n" + "=" * 60)
    print("Document Analysis Examples")
    print("=" * 60)
    
    print("\nPDF Analysis:")
    print("-" * 60)
    print("(Skipped - requires local PDF file)")
    print("To use: analyzer.analyze_document('path/to/document.pdf', 'Summarize this document')")
    
    print("\nExample use cases:")
    print("- Extract structured data from invoices")
    print("- Summarize research papers")
    print("- Answer questions about contracts")
    print("- Convert tables to JSON")


def demo_multimodal_conversation():
    """Demonstrate combining text and images in conversation."""
    print("\n" + "=" * 60)
    print("Multimodal Conversation")
    print("=" * 60)
    
    print("\nCombining Multiple Modalities:")
    print("-" * 60)
    print("You can mix text, images, audio, video, and documents in a single conversation:")
    print("""
Example workflow:
1. User uploads an image of a recipe
2. Bot describes the dish
3. User asks for nutritional information
4. Bot provides detailed nutrition facts
5. User uploads audio asking for substitutions
6. Bot suggests alternatives
    """)


if __name__ == "__main__":
    demo_image_analysis()
    demo_audio_analysis()
    demo_video_analysis()
    demo_document_analysis()
    demo_multimodal_conversation()
    
    print("\n" + "=" * 60)
    print("Setup Instructions")
    print("=" * 60)
    print("""
To run these examples with actual files:

1. Place your media files in a 'media' folder:
   - media/sample_image.png
   - media/sample_audio.wav
   - media/sample_video.mp4
   - media/sample_document.pdf

2. Update the code to use these paths:
   analyzer.analyze_image_from_file('media/sample_image.png', 'Describe this')

3. For large files (>10MB), use FilesAPIAnalyzer:
   files_analyzer = FilesAPIAnalyzer()
   files_analyzer.analyze_large_file('media/large_video.mp4', 'Summarize')
    """)
