"""Forms for the detector app."""
from django import forms


class NewsForm(forms.Form):
    """Single-field form for submitting a news statement."""

    statement = forms.CharField(
        label='',
        widget=forms.Textarea(attrs={
            'id': 'news-statement',
            'rows': 6,
            'placeholder': (
                'Paste or type a news headline or statement here…\n\n'
                'e.g. "Scientists discover a new cure for cancer using common household items."'
            ),
            'class': 'news-textarea',
            'spellcheck': 'true',
            'autofocus': True,
        }),
        min_length=10,
        max_length=5000,
        error_messages={
            'required': 'Please enter a news statement to analyze.',
            'min_length': 'Statement must be at least 10 characters long.',
        },
    )
