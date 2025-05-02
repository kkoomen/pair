def load_system_prompt_template(filename):
    return open(f"system_prompt_templates/{filename}.txt", "r", encoding="utf-8").read()


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        hours = int(seconds // 3600)
        minutes = int(seconds // 60) % 60
        remaining_seconds = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {remaining_seconds:.2f}s"

        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def pluralize(n, singular: str, plural: str) -> str:
    suffix = singular if n == 1 else plural
    return f"{n} {suffix}"
