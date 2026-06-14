(function () {
    const accuracy = window.dashboardData?.accuracy ?? null;
    const precision = window.dashboardData?.precision ?? null;
    const recall = window.dashboardData?.recall ?? null;
    const f1 = window.dashboardData?.f1 ?? null;

    if (accuracy === null || precision === null || recall === null || f1 === null) {
        return;
    }

    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                label: 'Score (%)',
                data: [accuracy * 100, precision * 100, recall * 100, f1 * 100],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.72)',
                    'rgba(255, 205, 86, 0.75)',
                    'rgba(54, 162, 235, 0.75)',
                    'rgba(255, 99, 132, 0.72)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 205, 86, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
})();
