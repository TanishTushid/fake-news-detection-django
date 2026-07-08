"""Admin registration for detector models."""
from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'label', 'confidence_percent', 'created_at', 'short_statement')
    list_filter = ('label', 'created_at')
    search_fields = ('statement',)
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def short_statement(self, obj):
        return obj.statement[:80] + ('...' if len(obj.statement) > 80 else '')
    short_statement.short_description = 'Statement'
