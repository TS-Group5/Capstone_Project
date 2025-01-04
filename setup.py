from setuptools import setup, find_packages

setup(
    name="Video Resume generation",
    version="0.1.0",
    package_dir={"": "src"},  # Tell setuptools packages are under src
    packages=find_packages(where="src"),  # List all packages under src
    install_requires=[
        "mysql-connector-python",
        "pyrebase4",
        "streamlit>=1.24.0",
        "opencv-python",
        "imageio",  # Added missing dependency
        "moviepy==1.0.3",      # Added missing dependency
    ],
    
    python_requires=">=3.8",

    # Add entry points for command line scripts
    entry_points={
        "console_scripts": [
            "generate-summary=generate_summary:main",
        ],
    },
)
