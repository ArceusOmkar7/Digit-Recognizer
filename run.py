#!/usr/bin/env python3
"""
Run script for MNIST Digit Recognizer FastAPI application
"""

import os
import sys
import subprocess


def check_virtual_environment():
    """Check if virtual environment is activated"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
        return True
    else:
        print("⚠ Warning: No virtual environment detected")
        print("  Consider activating the virtual environment: env\\Scripts\\activate")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import joblib
        import numpy
        import PIL
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install dependencies with: pip install -r requirements.txt")
        return False


def check_model_file():
    """Check if the model file exists"""
    model_path = os.path.join("models", "random_forest_clf.joblib")
    if os.path.exists(model_path):
        print("✓ Model file found")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Make sure the model file is properly tracked with Git LFS")
        return False


def run_server():
    """Run the FastAPI server"""
    try:
        print("\n🚀 Starting MNIST Digit Recognizer server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📱 Web interface at: http://localhost:8000")
        print("📋 API docs at: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop the server\n")

        # Import and run the app
        from app.main import app
        import uvicorn

        # Check if running in production (Render sets PORT environment variable)
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0" if port != 8000 else "127.0.0.1"

        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False if port != 8000 else True,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        sys.exit(1)


def main():
    """Main function to run the application"""
    print("🤖 MNIST Digit Recognizer - Starting Application")
    print("=" * 50)

    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run checks
    check_virtual_environment()

    if not check_dependencies():
        sys.exit(1)

    if not check_model_file():
        sys.exit(1)

    # Run the server
    run_server()


if __name__ == "__main__":
    main()
