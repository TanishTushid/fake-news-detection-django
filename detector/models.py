"""
Django models for the detector app.
Stores every prediction made by users for history and analytics.
"""
from django.db import models


class Prediction(models.Model):
    """Represents a single fake-news detection request and its result."""

    LABEL_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Fake'),
    ]

    statement = models.TextField(help_text="The news statement submitted by the user.")
    label = models.CharField(
        max_length=4,
        choices=LABEL_CHOICES,
        help_text="Predicted label: REAL or FAKE.",
    )
    confidence = models.FloatField(
        help_text="Model confidence score (0.0 – 1.0)."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the prediction was made.",
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'

    def __str__(self):
        return f"[{self.label}] {self.statement[:60]}..."

    @property
    def confidence_percent(self):
        """Return confidence as a rounded percentage string."""
        return f"{self.confidence * 100:.1f}%"

    @property
    def is_fake(self):
        return self.label == 'FAKE'
