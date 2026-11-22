"""Backward-compatible imports; implementation lives in chapter02_tokenization.dataset_and_dataloader."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("chapter02_tokenization.dataset_and_dataloader", run_name="__main__")
else:
    from chapter02_tokenization.dataset_and_dataloader import *  # noqa: F401,F403
