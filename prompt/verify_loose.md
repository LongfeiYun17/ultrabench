You are an evaluator of constraint satisfaction in generated text. 

Given a set of constraints and a generated text, determine for each constraint whether it is reasonably and substantially satisfied based on the content of the text.

Do not guess or infer missing information. Base your judgment strictly on what is explicitly present in the text.

If a constraint is substantially fulfilled, answer "YES".
If a constraint is clearly unmet or seriously deficient, answer "NO".

Output exactly {number} lines.
Each line must contain only "YES" or "NO", corresponding to each constraint in order.
Do not output any explanation or additional text. Only the sequence of "YES"/"NO" answers.

---
Generated Text:
{text}

Constraints:
{constraints}
---
Answer:
