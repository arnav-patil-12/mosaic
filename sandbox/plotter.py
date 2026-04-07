import matplotlib.pyplot as plt

benchmarks = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

hard_macros = [246, 271, 290, 295, 178, 291, 301, 253, 786, 373, 651, 424, 614, 393, 458, 760, 285]
soft_macros = [894, 1075, 1148, 1085, 900, 1040, 1030, 1048, 1982, 1195, 1985, 1301, 1529, 1138, 1315, 1844, 1029]
total_macros = [hard_macros[i] + soft_macros[i] for i in range(17)]

total_nets = [7269, 10944, 10247, 13555, 14085, 19920, 20694, 18341, 39140, 25273, 40996, 28201, 57333, 46467, 67118, 83132, 53763]
avg_nets = [total_nets[i] / total_macros[i] for i in range(17)]

# Bar width
bar_width = 0.25

# Positions for the bars
r1 = [x - bar_width for x in range(len(benchmarks))]
r2 = [x for x in range(len(benchmarks))]
r3 = [x + bar_width for x in range(len(benchmarks))]

plt.figure(figsize=(12, 6))
plt.bar(r1, hard_macros, width=bar_width, label="Hard Macros")
plt.bar(r2, soft_macros, width=bar_width, label="Soft Macros")
plt.bar(r3, total_macros, width=bar_width, label="Total Macros")

plt.xlabel("Benchmark", fontsize=12)
plt.ylabel("Number of Macros", fontsize=12)
plt.xticks(range(len(benchmarks)), benchmarks, fontsize=10)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"sandbox/poiqweur.png", dpi=600, bbox_inches="tight")

print(len(total_nets))