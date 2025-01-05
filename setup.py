from setuptools import setup, find_packages

setup(
    name="Video Resume generation",
    version="0.1.0",
    install_requires=[
        "mysql-connector-python",
        "pyrebase4",
        "streamlit>=1.24.0",
        "opencv-python",
        "imageio",  # Added missing dependency
        "moviepy==1.0.3",      # Added missing dependency
    ],
    
    python_requires=">=3.8",   
)
