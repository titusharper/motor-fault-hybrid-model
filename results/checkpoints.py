if early.best_state is not None:
    model.load_state_dict(early.best_state)

ckpt_path = REPO_ROOT / "results" / "checkpoints" / "best_hybrid_film.pth"
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), ckpt_path)
print(f"Best model saved to {ckpt_path}")
