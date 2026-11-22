"""Backward-compatible imports; implementation lives in chapter04_gpt_model.gpt_model."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("chapter04_gpt_model.gpt_model", run_name="__main__")
else:
    from chapter04_gpt_model.gpt_model import *  # noqa: F401,F403
