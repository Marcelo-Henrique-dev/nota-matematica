-- SOR: Estrutura da tabela para os dados brutos dos estudantes
CREATE TABLE sor_students (
    id INT PRIMARY KEY,
    gender VARCHAR(20),
    race_ethnicity VARCHAR(50),
    parental_level_of_education VARCHAR(50),
    lunch VARCHAR(20),
    test_preparation_course VARCHAR(20),
    math_score INT,
    reading_score INT,
    writing_score INT
);