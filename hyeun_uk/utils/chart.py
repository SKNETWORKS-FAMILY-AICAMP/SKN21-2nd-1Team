import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



class ChartMaker:
    def __init__(self,
                 df: pd.DataFrame,
                 target: str = "churn"):

        self.df = df.copy()
        self.target = target
        self.colors = ["#0f18b5", "#9ea30a"]






        # BAR,PIE-CHART GRAPH
    def bar_pie(self,
                figsize=(14, 5),
                explode=[0, 0.1]):
        """
        하나의 Figure 안에 Bar + Pie 동시에 출력
        """
        target_counts = self.df[self.target].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=figsize)


        # AXES[0] BAR-CHART GRAPH
        axes[0].bar(
            target_counts.index,
            target_counts.values,
            color=self.colors
        )
        axes[0].set_title("BAR-GRAPH", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("1:해지, 0:유지", fontsize=10)
        axes[0].set_ylabel("인원", fontsize=10)
        axes[0].grid(axis="y", alpha=0.5)


        # AXES[1] PIE CHART GRAPH
        axes[1].pie(
            target_counts.values,
            labels=target_counts.index,
            autopct="%1.2f%%",
            colors=self.colors,
            startangle=90,
            explode=explode,
            shadow=False,
            textprops={"fontsize": 10}
        )
        axes[1].set_title("PIE-GRAPH", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.show()






        # BOXPLOT-CHART GRAPH
    def boxplot(self,
                cols: list = None,
                rows=2,
                cols_per_row=3,
                figsize=(18, 10)):

        df = self.df.copy()

        
        if cols is None:
            
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != self.target]
            default_cols = [
                "age", "balance", "credit_score",
                "tenure", "estimated_salary", "products_number"
            ]

            
            cols = [c for c in default_cols if c in numeric_cols]

            if len(cols) == 0:
                cols = numeric_cols[:6]


        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target]
        cols = [col for col in cols if col in numeric_cols]

        if not cols:
            raise ValueError("박스플롯을 그릴 수 있는 numeric 컬럼이 없습니다.")

        total_plots = rows * cols_per_row

        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
        axes = axes.flatten()

        for idx, col in enumerate(cols[:total_plots]):
            sns.boxplot(
                data=df,
                x=self.target,
                y=col,
                ax=axes[idx],
                palette=self.colors,
                hue=self.target,
                dodge=False,
                legend=False
            )

            axes[idx].set_title(f"{col.upper()}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel(self.target, fontsize=10)
            axes[idx].set_ylabel(col, fontsize=10)
            axes[idx].grid(axis="y", alpha=0.4)

        
        for idx in range(len(cols), total_plots):
            axes[idx].set_visible(False)

        fig.suptitle("Multi Boxplot", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()






        # HEATMAP-CHART GRAPH
    def heatmap(self,
            cols: list = None,
            figsize=(12, 10),
            method: str = "pearson",
            annot: bool = True,
            round_decimals: int = 2):
        

        df = self.df.copy()

        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        
        if cols is None:
            cols = numeric_cols
        else:
            
            cols = [c for c in cols if c in numeric_cols]

        if len(cols) == 0:
            raise ValueError("heatmap에 사용할 수 있는 수치형 컬럼이 없습니다.")

        corr = df[cols].corr(method=method).round(round_decimals)

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            annot=annot,
            fmt=f".{round_decimals}f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar=True
        )
        plt.title(f"Correlation Heatmap ({method})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()






        # HIST-CHART GRAPH
    def hist(self,
         cols: list = None,
         rows=2,
         cols_per_row=3,
         bins=30,
         figsize=(18, 10)):
       

        df = self.df.copy()

       
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target]

        if cols is None:
            
            default_cols = [
                "age", "balance", "credit_score",
                "tenure", "estimated_salary", "products_number"
            ]
            cols = [c for c in default_cols if c in numeric_cols]

            if len(cols) == 0:
                cols = numeric_cols[:6]  # fallback

        
        cols = [c for c in cols if c in numeric_cols]

        if len(cols) == 0:
            raise ValueError("유효한 수치형 컬럼이 없습니다.")

        
        total_plots = rows * cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
        axes = axes.flatten()

        targets = sorted(df[self.target].unique())

        
        for idx, col in enumerate(cols[:total_plots]):
            ax = axes[idx]

            for target, color in zip(targets, self.colors):
                data = df.loc[df[self.target] == target, col]

                ax.hist(
                    data,
                    bins=bins,
                    alpha=0.6,
                    edgecolor="black",
                    color=color,
                    label=f"{self.target}={target}",
                )

            ax.set_title(f"{col.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend()
            ax.grid(axis="y", alpha=0.4)

        
        for idx in range(len(cols), total_plots):
            axes[idx].set_visible(False)

        fig.suptitle("Histogram Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()





print("chart 로드 완료")