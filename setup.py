from setuptools import setup, find_packages

setup(
    name="Video Resume generation",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": [
            "img/*.png",
            "*.mp4",
            "src/video/*.mp4",
            "src/merged_video/*.mp4",
            "src/final_video/*.mp4"
        ],
    },
    py_modules=["1_🏠_Home"],
    data_files=[
        ('pages', ['pages/2_🔒_Login.py', 'pages/3_⬆️_Upload_Resume.py']),
        ('img', ['img/back.png', 'img/login.png', 'img/resume.png']),
        ('', [
            'Skills.mp4', 'Achievement.mp4', 'Contact.mp4',
            'Goals.mp4', 'Experience.mp4'
        ]),
        ('src/video', [
            'src/video/Introduction.mp4', 'src/video/Skills.mp4',
            'src/video/Achievement.mp4', 'src/video/Contact.mp4',
            'src/video/Goals.mp4', 'src/video/Experience.mp4'
        ]),
        ('src/merged_video', [
            'src/merged_video/Introduction.mp4', 'src/merged_video/Skills.mp4',
            'src/merged_video/Achievement.mp4', 'src/merged_video/Contact.mp4',
            'src/merged_video/Goals.mp4', 'src/merged_video/Experience.mp4'
        ]),
        ('src/final_video', ['src/final_video/merged_video.mp4']),
    ],
    install_requires=[
        "mysql-connector-python",
        "pyrebase4",
        "streamlit>=1.24.0",
        "opencv-python",
        "imageio",
        "moviepy==1.0.3",
    ],
    python_requires=">=3.8",
)
