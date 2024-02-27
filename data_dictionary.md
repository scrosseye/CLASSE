- [Data dictionary](#data-dictionary)
  - [Summaries](#summaries)
  - [Prompts](#prompts)

## Data dictionary

### Summaries

| Column             | Definition                                                              |
| ------------------ | ------------------------------------------------------------------------|
| student_id         | The ID of the student writer                                            |
| prompt_id          | The ID of the prompt which links to the prompt file                     |
| text               | The full text of the student's summary                                  |
| content            | The content score for the summary                                       |
| wording            | The wording score for the summary                                       |
| student_grade      | The grade level of the student                                          |
| ell_class          | The ELL status of the student (0 = non-ELL, 1 = ELL)                    |
| split              | The competition split this summary is part of                           |

### Prompts

| Column             | Definition                                                              |
| ------------------ | ------------------------------------------------------------------------|
| prompt_id          | The ID of the prompt which links to the summaries file                  |
| prompt_question    | The specific question the students are asked to respond to              |
| prompt_title       | A short-hand title for the prompt                                       |
| prompt_url         | The URL from the prompt hosted on CommonLit's website                   |
| prompt_html        | The HTML of the full prompt text                                        |
| prompt_text        | The full prompt text without HTML elements                              |