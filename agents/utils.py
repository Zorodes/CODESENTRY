def build_code_context(code_chunks: list[dict]) -> str:
    lines = []
    for chunk in code_chunks:
        cid = f"[code:{chunk['id']}]"
        lines.append(
            f"{cid} {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']} "
            f"({chunk['language']} {chunk['chunk_type']}: {chunk['name']})\n"
            f"```{chunk['language']}\n{chunk['code']}\n```"
        )
    return "\n\n".join(lines)

def build_precedent_context(precedents: list[dict]) -> str:
    lines = []
    for p in precedents:
        pid = f"[precedent:{p['id']}]"
        lines.append(
            f"{pid} PR #{p['pr_number']} - {p['pr_title']}\n"
            f"File: {p['file_path']}\n"
            f"Diff hunk:\n```diff\n{p['diff_hunk']}\n```\n"
            f"Reviewer comment: {p['comment_body']}"
        )
    return "\n\n".join(lines)
