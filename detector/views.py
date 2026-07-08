"""
Views for the detector app.

Routes:
    /               → home      (GET: show form)
    /predict/       → predict   (POST: run inference, redirect to result)
    /result/<id>/   → result    (GET: show prediction detail)
    /history/       → history   (GET: paginated list of predictions)
    /about/         → about     (GET: static info page)
"""
import logging

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.views.decorators.http import require_http_methods

from .forms import NewsForm
from .models import Prediction
from .utils import predict as run_prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def home(request):
    """Render the main landing page with the news-submission form."""
    form = NewsForm()
    recent = Prediction.objects.all()[:5]
    total_checked = Prediction.objects.count()
    total_fake = Prediction.objects.filter(label='FAKE').count()
    total_real = Prediction.objects.filter(label='REAL').count()

    context = {
        'form': form,
        'recent': recent,
        'total_checked': total_checked,
        'total_fake': total_fake,
        'total_real': total_real,
    }
    return render(request, 'detector/home.html', context)


# ---------------------------------------------------------------------------
# Predict (POST only)
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def predict_view(request):
    """
    Accept a POST request with a news statement, run the ML pipeline,
    save the result to the database, and redirect to the result page.
    """
    form = NewsForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Please enter a valid news statement (at least 10 characters).")
        return redirect('home')

    raw_text = form.cleaned_data['statement']

    try:
        result = run_prediction(raw_text)
    except RuntimeError as exc:
        messages.error(request, str(exc))
        return redirect('home')
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        messages.error(request, "An unexpected error occurred. Please try again.")
        return redirect('home')

    prediction = Prediction.objects.create(
        statement=raw_text,
        label=result['label'],
        confidence=result['confidence'],
    )

    return redirect('result', pk=prediction.pk)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def result(request, pk):
    """Show the prediction result for a given Prediction id."""
    prediction = get_object_or_404(Prediction, pk=pk)
    similar_count = Prediction.objects.filter(label=prediction.label).count()

    context = {
        'prediction': prediction,
        'confidence_pct': round(prediction.confidence * 100, 1),
        'similar_count': similar_count,
    }
    return render(request, 'detector/result.html', context)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def history(request):
    """Paginated table of all past predictions with basic analytics."""
    qs = Prediction.objects.all()

    # Simple search filter
    query = request.GET.get('q', '').strip()
    if query:
        qs = qs.filter(statement__icontains=query)

    label_filter = request.GET.get('label', '').upper()
    if label_filter in ('FAKE', 'REAL'):
        qs = qs.filter(label=label_filter)

    paginator = Paginator(qs, 15)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    total = Prediction.objects.count()
    fake_count = Prediction.objects.filter(label='FAKE').count()
    real_count = Prediction.objects.filter(label='REAL').count()
    fake_pct = round(fake_count / total * 100, 1) if total else 0
    real_pct = round(real_count / total * 100, 1) if total else 0

    context = {
        'page_obj': page_obj,
        'query': query,
        'label_filter': label_filter,
        'total': total,
        'fake_count': fake_count,
        'real_count': real_count,
        'fake_pct': fake_pct,
        'real_pct': real_pct,
    }
    return render(request, 'detector/history.html', context)


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def about(request):
    """Render the static about page."""
    return render(request, 'detector/about.html')
