def calculate_metric(df, column, aggregation):

    if aggregation == "sum":
        return df[column].sum()

    elif aggregation == "mean":
        return df[column].mean()

    elif aggregation == "median":
        return df[column].median()

    elif aggregation == "count":
        return df[column].count()

    elif aggregation == "nunique":
        return df[column].nunique()

    elif aggregation == "min":
        return df[column].min()

    elif aggregation == "max":
        return df[column].max()


def calculate_kpis(df, kpis):

    results = []

    for kpi in kpis:

        kpi_type = kpi["type"]
        percentage = kpi.get("percentage", False)

        # simple KPI
        if kpi_type == "simple":

            value = calculate_metric(
                df,
                kpi["column"],
                kpi["aggregation"]
            )

        # derived KPI
        elif kpi_type == "ratio":

            numerator = calculate_metric(
                df,
                kpi["numerator"]["column"],
                kpi["numerator"]["aggregation"]
            )

            denominator = calculate_metric(
                df,
                kpi["denominator"]["column"],
                kpi["denominator"]["aggregation"]
            )

            if denominator != 0:
                value = numerator / denominator
            else:
                value = None

        else:
            value = None

        if percentage and value is not None:
            value = value * 100

        results.append({
            "name": kpi["name"],
            "value": value,
            "percentage": percentage
        })

    return results