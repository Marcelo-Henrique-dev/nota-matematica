# Modelagem dos Dados

## SOR (Source of Record)
- **Arquivo:** students.csv
- **Descrição:** Dados brutos dos estudantes.

## SOT (Source of Truth)
- **Objeto:** df_processed
- **Descrição:** Dados tratados e transformados via one-hot encoding.

## SPEC (Specification)
- **Features:**
  - reading score
  - writing score
  - parental level of education_associate's degree
  - parental level of education_bachelor's degree
  - parental level of education_high school
  - parental level of education_master's degree
  - parental level of education_some college
  - parental level of education_some high school
  - lunch_free/reduced
  - lunch_standard
  - test preparation course_completed
  - test preparation course_none
- **Target:** math score
- **Modelo:** LinearRegression (sklearn)