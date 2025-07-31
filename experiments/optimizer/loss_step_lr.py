from cs336_basics import model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

num_experiments = 10
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
results = {num: {"loss": []} for num in range(num_experiments)}

for num in range(num_experiments):
    tmp_weights = nn.Parameter(weights.clone())
    opt = model.SGD([tmp_weights], lr=num/8.0)
    
    for t in range(100):
        opt.zero_grad()
        loss = (tmp_weights ** 2).sum()
        loss.backward()
        opt.step()
        results[num]["loss"].append(loss.item())

plt.figure(figsize=(10, 6))
for num, date_dict in results.items():
    losses = date_dict["loss"]
    x_values = range(len(losses))
    plt.plot(x_values, losses, label=f"SGD lr={num/8.0:.2f}")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Curves for Different Learning Rates")
plt.legend()
output_filename = 'loss_step_lr.png'
plt.savefig(output_filename)
plt.close