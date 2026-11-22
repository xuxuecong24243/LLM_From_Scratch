"""Backward-compatible imports; implementation lives in chapter03_attention.multi_head_attention."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("chapter03_attention.multi_head_attention", run_name="__main__")
else:
    from chapter03_attention.multi_head_attention import *  # noqa: F401,F403
