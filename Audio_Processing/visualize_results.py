"""
Results Visualization
Create visualizations of music listening patterns and genre predictions
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_listening_patterns(streaming_df, output_dir):
    """
    Create plots of listening patterns.

    Args:
        streaming_df (pd.DataFrame): Streaming history data
        output_dir (Path): Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nCreating listening pattern visualizations...")

    # Set style
    plt.style.use("seaborn")

    # 1. Daily listening time
    daily_listening = streaming_df.groupby("date")["duration_min"].sum().reset_index()

    fig = px.line(
        daily_listening,
        x="date",
        y="duration_min",
        title="Daily Listening Time",
        labels={"date": "Date", "duration_min": "Listening Time (minutes)"},
    )
    fig.write_html(output_dir / "daily_listening.html")

    # 2. Hourly patterns
    hourly = streaming_df.groupby("hour")["duration_min"].mean().reset_index()

    fig = px.bar(
        hourly,
        x="hour",
        y="duration_min",
        title="Average Listening Time by Hour",
        labels={"hour": "Hour of Day", "duration_min": "Average Minutes"},
    )
    fig.write_html(output_dir / "hourly_patterns.html")

    # 3. Day of week patterns
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly = (
        streaming_df.groupby("day_of_week")["duration_min"]
        .mean()
        .reindex(day_order)
        .reset_index()
    )

    fig = px.bar(
        weekly,
        x="day_of_week",
        y="duration_min",
        title="Average Listening Time by Day",
        labels={"day_of_week": "Day of Week", "duration_min": "Average Minutes"},
    )
    fig.write_html(output_dir / "weekly_patterns.html")

    # 4. Top artists
    top_artists = (
        streaming_df.groupby("artist_name")["duration_min"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        top_artists,
        x="duration_min",
        y="artist_name",
        orientation="h",
        title="Top 15 Most Listened Artists",
        labels={"artist_name": "Artist", "duration_min": "Total Minutes"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(output_dir / "top_artists.html")

    # 5. Monthly patterns
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly = (
        streaming_df.groupby(["year", "month"])["duration_min"].sum().reset_index()
    )
    monthly["month_year"] = monthly["month"] + " " + monthly["year"].astype(str)

    fig = px.bar(
        monthly,
        x="month_year",
        y="duration_min",
        title="Monthly Listening Time",
        labels={"month_year": "Month", "duration_min": "Total Minutes"},
    )
    fig.write_html(output_dir / "monthly_patterns.html")

    # 6. Listening behavior
    behavior_metrics = pd.DataFrame(
        {
            "Metric": ["Skipped", "Shuffle On", "Offline Mode"],
            "Percentage": [
                (streaming_df["skipped"].mean() * 100),
                (streaming_df["shuffle"].mean() * 100),
                (streaming_df["offline"].mean() * 100),
            ],
        }
    )

    fig = px.bar(
        behavior_metrics,
        x="Metric",
        y="Percentage",
        title="Listening Behavior",
        labels={"Percentage": "Percentage of Tracks"},
    )
    fig.write_html(output_dir / "listening_behavior.html")

    print("Listening pattern visualizations created!")


def plot_genre_predictions(predictions_df, output_dir):
    """
    Create plots of genre predictions.

    Args:
        predictions_df (pd.DataFrame): Prediction results
        output_dir (Path): Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nCreating genre prediction visualizations...")

    # 1. Genre distribution
    genre_dist = predictions_df["predicted_genre"].value_counts().reset_index()
    genre_dist.columns = ["Genre", "Count"]

    fig = px.pie(
        genre_dist,
        values="Count",
        names="Genre",
        title="Distribution of Predicted Genres",
    )
    fig.write_html(output_dir / "genre_distribution.html")

    # 2. Genre confidence
    prob_cols = [col for col in predictions_df.columns if col.startswith("prob_")]
    genres = [col.replace("prob_", "") for col in prob_cols]

    conf_data = []
    for genre in genres:
        conf_data.append(
            {
                "Genre": genre,
                "Average Confidence": predictions_df[f"prob_{genre}"].mean() * 100,
            }
        )

    conf_df = pd.DataFrame(conf_data)

    fig = px.bar(
        conf_df,
        x="Genre",
        y="Average Confidence",
        title="Average Prediction Confidence by Genre",
        labels={"Average Confidence": "Average Confidence (%)"},
    )
    fig.write_html(output_dir / "genre_confidence.html")

    # 3. Genre by time of day
    hourly_genre = (
        predictions_df.groupby(["hour", "predicted_genre"]).size().unstack(fill_value=0)
    )

    # Convert to percentages
    hourly_genre = hourly_genre.div(hourly_genre.sum(axis=1), axis=0) * 100

    fig = px.area(
        hourly_genre.reset_index(),
        x="hour",
        y=hourly_genre.columns,
        title="Genre Distribution by Hour",
        labels={"hour": "Hour of Day", "value": "Percentage"},
    )
    fig.write_html(output_dir / "genre_by_hour.html")

    # 4. Genre by day of week
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily_genre = (
        predictions_df.groupby(["day_of_week", "predicted_genre"])
        .size()
        .unstack(fill_value=0)
    )
    daily_genre = daily_genre.reindex(day_order)

    # Convert to percentages
    daily_genre = daily_genre.div(daily_genre.sum(axis=1), axis=0) * 100

    fig = px.area(
        daily_genre.reset_index(),
        x="day_of_week",
        y=daily_genre.columns,
        title="Genre Distribution by Day",
        labels={"day_of_week": "Day of Week", "value": "Percentage"},
    )
    fig.write_html(output_dir / "genre_by_day.html")

    # 5. Top artists by genre
    artist_genres = []
    for genre in genres:
        # Get top artists for this genre
        genre_tracks = predictions_df[predictions_df["predicted_genre"] == genre]
        top_artists = (
            genre_tracks.groupby("artist_name")["duration_min"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        for artist, duration in top_artists.items():
            artist_genres.append(
                {"Genre": genre, "Artist": artist, "Minutes": duration}
            )

    artist_genre_df = pd.DataFrame(artist_genres)

    fig = px.bar(
        artist_genre_df,
        x="Minutes",
        y="Artist",
        color="Genre",
        orientation="h",
        title="Top Artists by Genre",
        labels={"Minutes": "Total Listening Time (minutes)"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(output_dir / "top_artists_by_genre.html")

    # 6. Genre trends over time
    monthly_genre = (
        predictions_df.groupby(["year", "month", "predicted_genre"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_genre = monthly_genre.div(monthly_genre.sum(axis=1), axis=0) * 100

    # Create month-year labels
    month_labels = [f"{row[1]} {row[0]}" for row in monthly_genre.index]

    fig = px.area(
        monthly_genre.reset_index(),
        x=month_labels,
        y=monthly_genre.columns,
        title="Genre Trends Over Time",
        labels={"x": "Month", "value": "Percentage"},
    )
    fig.write_html(output_dir / "genre_trends.html")

    print("Genre prediction visualizations created!")


def create_dashboard(streaming_df, predictions_df, output_dir):
    """
    Create an interactive dashboard combining all visualizations.

    Args:
        streaming_df (pd.DataFrame): Streaming history data
        predictions_df (pd.DataFrame): Prediction results
        output_dir (Path): Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nCreating interactive dashboard...")

    # Create subplot grid
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Daily Listening Time",
            "Genre Distribution",
            "Top Artists",
            "Genre by Hour",
            "Listening Behavior",
            "Genre Confidence",
        ),
    )

    # 1. Daily listening time
    daily = streaming_df.groupby("date")["duration_min"].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=daily["date"], y=daily["duration_min"], name="Daily Minutes"),
        row=1,
        col=1,
    )

    # 2. Genre distribution
    genre_dist = predictions_df["predicted_genre"].value_counts()
    fig.add_trace(
        go.Pie(labels=genre_dist.index, values=genre_dist.values, name="Genres"),
        row=1,
        col=2,
    )

    # 3. Top artists
    top_artists = (
        streaming_df.groupby("artist_name")["duration_min"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    fig.add_trace(
        go.Bar(
            x=top_artists.values, y=top_artists.index, orientation="h", name="Artists"
        ),
        row=2,
        col=1,
    )

    # 4. Genre by hour
    hourly_genre = (
        predictions_df.groupby(["hour", "predicted_genre"]).size().unstack(fill_value=0)
    )
    hourly_genre = hourly_genre.div(hourly_genre.sum(axis=1), axis=0) * 100

    for genre in hourly_genre.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_genre.index,
                y=hourly_genre[genre],
                name=genre,
                stackgroup="one",
            ),
            row=2,
            col=2,
        )

    # 5. Listening behavior
    behavior_metrics = pd.DataFrame(
        {
            "Metric": ["Skipped", "Shuffle On", "Offline Mode"],
            "Percentage": [
                (streaming_df["skipped"].mean() * 100),
                (streaming_df["shuffle"].mean() * 100),
                (streaming_df["offline"].mean() * 100),
            ],
        }
    )
    fig.add_trace(
        go.Bar(
            x=behavior_metrics["Metric"],
            y=behavior_metrics["Percentage"],
            name="Behavior",
        ),
        row=3,
        col=1,
    )

    # 6. Genre confidence
    prob_cols = [col for col in predictions_df.columns if col.startswith("prob_")]
    genres = [col.replace("prob_", "") for col in prob_cols]

    conf_data = []
    for genre in genres:
        conf_data.append(
            {"Genre": genre, "Confidence": predictions_df[f"prob_{genre}"].mean() * 100}
        )

    conf_df = pd.DataFrame(conf_data)
    fig.add_trace(
        go.Bar(x=conf_df["Genre"], y=conf_df["Confidence"], name="Confidence"),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=1200, showlegend=True, title_text="Music Listening Analysis Dashboard"
    )

    # Save dashboard
    dashboard_path = output_dir / "dashboard.html"
    fig.write_html(dashboard_path)
    print(f"Dashboard saved to: {dashboard_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument(
        "--streaming",
        type=str,
        default="Ingested_Data/cleaned_streaming_history.csv",
        help="Path to cleaned streaming history",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="Predictions/genre_predictions.csv",
        help="Path to genre predictions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Visualizations",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.streaming):
        print(f"Error: Streaming history not found: {args.streaming}")
        return

    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        return

    # Load data
    print("Loading data...")
    streaming_df = pd.read_csv(args.streaming)
    predictions_df = pd.read_csv(args.predictions)

    # Create visualizations
    plot_listening_patterns(streaming_df, args.output)
    plot_genre_predictions(predictions_df, args.output)
    create_dashboard(streaming_df, predictions_df, args.output)

    print("\nVisualization creation complete!")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    main()
