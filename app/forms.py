"""Flask-WTF forms used in the application."""

from __future__ import annotations

from typing import List, Tuple

from flask_wtf import FlaskForm
from wtforms import FileField, PasswordField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Optional, Regexp
from flask_wtf.file import FileAllowed


class PredictionForm(FlaskForm):
    tweet_text = TextAreaField(
        "Tweet Text",
        validators=[
            Optional(),
            Length(min=10, max=500, message="Tweet must be between 10 and 500 characters."),
            Regexp(
                r"^[a-zA-Z0-9\s\.,!?@#\-\'\"]+$",
                message="Tweet contains unsupported characters.",
            ),
        ],
        render_kw={"rows": 4, "placeholder": "Paste the tweet you want to analyze."},
    )
    batch_file = FileField(
        "or Upload CSV",
        validators=[Optional(), FileAllowed(["csv"], "CSV files only.")],
    )
    airline = SelectField(
        "Airline (optional)",
        choices=[],
        validators=[Optional()],
    )
    submit = SubmitField("Analyze Sentiment")

    def set_airline_choices(self, airlines: List[str]) -> None:
        choices: List[Tuple[str, str]] = [("", "-- Not Specified --")]
        choices.extend((value, value) for value in airlines)
        self.airline.choices = choices

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators=extra_validators):
            return False

        text_present = bool(self.tweet_text.data and self.tweet_text.data.strip())
        file_present = bool(self.batch_file.data and getattr(self.batch_file.data, 'filename', ''))

        if not text_present and not file_present:
            error_message = "Provide tweet text or upload a CSV file."
            self.tweet_text.errors.append(error_message)
            return False

        if text_present and file_present:
            error_message = "Choose either a single tweet or a CSV upload, not both."
            self.batch_file.errors.append(error_message)
            return False

        return True


class AdminLoginForm(FlaskForm):
    username = StringField("Admin Username", validators=[DataRequired()])
    password = PasswordField("Admin Password", validators=[DataRequired()])
    submit = SubmitField("Login")
