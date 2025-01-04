CREATE TABLE students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    class VARCHAR(10),
    reg_no VARCHAR(20) UNIQUE,
    password VARCHAR(255)
);

CREATE TABLE questions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    subject VARCHAR(50),
    level ENUM('easy', 'intermediate', 'difficult'),
    question_text TEXT,
    option_a VARCHAR(255),
    option_b VARCHAR(255),
    option_c VARCHAR(255),
    option_d VARCHAR(255),
    correct_option CHAR(1)
);

CREATE TABLE quiz_sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT,
    question_id INT,
    selected_option CHAR(1),
    exam_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
);
