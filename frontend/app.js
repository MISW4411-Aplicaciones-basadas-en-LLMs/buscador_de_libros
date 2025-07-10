document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const queryInput = document.getElementById('query');
    const thresholdInput = document.getElementById('threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const resultsContainer = document.getElementById('results');
    const loader = document.getElementById('loader');
    const noResults = document.getElementById('no-results');

    // Update threshold value display
    thresholdInput.addEventListener('input', () => {
        thresholdValue.textContent = parseFloat(thresholdInput.value).toFixed(2);
    });

    // Search button event listener
    searchBtn.addEventListener('click', searchBooks);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchBooks();
        }
    });

    async function searchBooks() {
        const query = queryInput.value.trim();
        const threshold = parseFloat(thresholdInput.value);

        if (!query) {
            alert('Por favor, introduce un término de búsqueda.');
            return;
        }

        // Show loader and clear previous results
        loader.style.display = 'block';
        resultsContainer.innerHTML = '';
        noResults.style.display = 'none';

        try {
            const response = await fetch('http://localhost:8000/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, threshold }),
            });

            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }

            const data = await response.json();
            displayBooks(data.results);
            displayPlot(data.results, data.query_embedding_2d);

        } catch (error) {
            console.error('Error al buscar libros:', error);
            resultsContainer.innerHTML = `<p class="error">Ocurrió un error al conectar con la API. Por favor, inténtalo de nuevo más tarde.</p>`;
        } finally {
            loader.style.display = 'none';
        }
    }

    function displayPlot(books, queryEmbedding) {
        const plotContainer = document.getElementById('plot-container');
        const plotDiv = document.getElementById('embedding-plot');
        const tooltipDiv = d3.select("#plot-tooltip");
        const distanceInfo = d3.select("#distance-info");

        if (!plotDiv || !plotContainer) return;

        if (books.length === 0) {
            plotContainer.style.display = 'none';
            return;
        }
        plotContainer.style.display = 'block';
        d3.select(plotDiv).selectAll("*").remove(); // Clear previous plot

        const margin = { top: 20, right: 20, bottom: 40, left: 40 };
        const width = plotDiv.clientWidth - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = d3.select(plotDiv)
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const plotData = books.map(b => ({
            x: b.book.embedding[0],
            y: b.book.embedding[1],
            label: b.book.title,
            type: 'book',
            score: b.score // Store the original score
        }));
        plotData.push({
            x: queryEmbedding[0],
            y: queryEmbedding[1],
            label: 'Tu consulta',
            type: 'query'
        });

        const xExtent = d3.extent(plotData, d => d.x);
        const yExtent = d3.extent(plotData, d => d.y);
        const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
        const yPadding = (yExtent[1] - yExtent[0]) * 0.1;

        const xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([height, 0]);

        const xAxis = d3.axisBottom(xScale).tickFormat(d3.format(".2f"));
        const yAxis = d3.axisLeft(yScale).tickFormat(d3.format(".2f"));

        const gX = svg.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(xAxis);

        const gY = svg.append("g")
            .call(yAxis);

        const clip = svg.append("defs").append("svg:clipPath")
            .attr("id", "clip")
            .append("svg:rect")
            .attr("width", width)
            .attr("height", height)
            .attr("x", 0)
            .attr("y", 0);

        const scatter = svg.append('g').attr("clip-path", "url(#clip)");

        let selectedPoints = [];

        const points = scatter.selectAll("circle")
            .data(plotData)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", d => d.type === 'query' ? 8 : 6)
            .style("fill", d => d.type === 'query' ? "#eab308" : "#3b82f6")
            .style("stroke", d => d.type === 'query' ? "#ca8a04" : "#2563eb")
            .style("stroke-width", 1.5)
            .style("cursor", "pointer")
            .on("mouseover", (event, d) => {
                tooltipDiv.transition().duration(200).style("opacity", .9);
                tooltipDiv.html(d.label)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                tooltipDiv.transition().duration(500).style("opacity", 0);
            })
            .on("click", function(event, d) {
                const point = d3.select(this);
                const isSelected = point.classed("selected");

                if (isSelected) {
                    point.classed("selected", false).style("stroke", d.type === 'query' ? "#ca8a04" : "#2563eb").style("stroke-width", 1.5);
                    selectedPoints = selectedPoints.filter(p => p.label !== d.label);
                } else {
                    if (selectedPoints.length >= 2) {
                        // Deselect all points before selecting a new one
                        points.classed("selected", false).style("stroke", p => p.type === 'query' ? "#ca8a04" : "#2563eb").style("stroke-width", 1.5);
                        selectedPoints = [];
                    }
                    point.classed("selected", true).style("stroke", "#e11d48").style("stroke-width", 3);
                    selectedPoints.push(d);
                }
                updateDistanceLine();
            });

        function updateDistanceLine() {
            svg.select(".distance-line").remove();
            distanceInfo.html(""); // Clear info by default

            if (selectedPoints.length === 2) {
                const [p1, p2] = selectedPoints;
                
                // Draw the line
                svg.append("line")
                    .attr("class", "distance-line")
                    .attr("x1", xScale(p1.x))
                    .attr("y1", yScale(p1.y))
                    .attr("x2", xScale(p2.x))
                    .attr("y2", yScale(p2.y))
                    .attr("stroke", "#e11d48")
                    .attr("stroke-width", 2)
                    .attr("stroke-dasharray", "5,5");

                // Calculate and display distances
                const dist2D = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
                let infoHTML = `<strong>Distancia Euclidiana:</strong> ${dist2D.toFixed(4)} <em>(en el gráfico 2D)</em>`;

                // Check if one point is the query and the other is a book to show API score
                const queryPoint = selectedPoints.find(p => p.type === 'query');
                const bookPoint = selectedPoints.find(p => p.type === 'book');

                if (queryPoint && bookPoint) {
                    infoHTML += `<br><strong>Similitud real (API):</strong> ${Math.round(bookPoint.score * 100)}%`;
                }
                
                distanceInfo.html(infoHTML);
            }
        }

        const zoom = d3.zoom()
            .scaleExtent([0.5, 20])
            .extent([[0, 0], [width, height]])
            .on("zoom", (event) => {
                const newXScale = event.transform.rescaleX(xScale);
                const newYScale = event.transform.rescaleY(yScale);

                gX.call(xAxis.scale(newXScale));
                gY.call(yAxis.scale(newYScale));

                points.attr('cx', d => newXScale(d.x))
                      .attr('cy', d => newYScale(d.y));
                
                if (selectedPoints.length === 2) {
                    const [p1, p2] = selectedPoints;
                    svg.select(".distance-line")
                        .attr("x1", newXScale(p1.x))
                        .attr("y1", newYScale(p1.y))
                        .attr("x2", newXScale(p2.x))
                        .attr("y2", newYScale(p2.y));
                }
            });

        const zoomBehavior = d3.zoom()
            .scaleExtent([0.5, 20])
            .extent([[0, 0], [width, height]])
            .on("zoom", (event) => {
                const newTransform = event.transform;
                const newXScale = newTransform.rescaleX(xScale);
                const newYScale = newTransform.rescaleY(yScale);

                gX.call(xAxis.scale(newXScale));
                gY.call(yAxis.scale(newYScale));

                points.attr('cx', d => newXScale(d.x))
                      .attr('cy', d => newYScale(d.y));
                
                if (selectedPoints.length === 2) {
                    const [p1, p2] = selectedPoints;
                    svg.select(".distance-line")
                        .attr("x1", newXScale(p1.x))
                        .attr("y1", newYScale(p1.y))
                        .attr("x2", newXScale(p2.x))
                        .attr("y2", newYScale(p2.y));
                }
            });

        const svgElement = d3.select(plotDiv).select("svg");
        svgElement.call(zoomBehavior);

        d3.select("#reset-plot-btn").on("click", () => {
            // Reset zoom and pan
            svgElement.transition().duration(750).call(zoomBehavior.transform, d3.zoomIdentity);
            
            // Clear selections
            points.classed("selected", false).style("stroke", p => p.type === 'query' ? "#ca8a04" : "#2563eb").style("stroke-width", 1.5);
            selectedPoints = [];
            updateDistanceLine();
        });
    }

    function displayBooks(books) {
        resultsContainer.innerHTML = '';

        if (books.length === 0) {
            noResults.style.display = 'block';
            return;
        }
        
        const resultsTitle = document.createElement('h2');
        resultsTitle.textContent = 'Resultados de la búsqueda';
        resultsTitle.classList.add('results-title');
        resultsContainer.appendChild(resultsTitle);

        books.forEach(book => {
            const bookElement = document.createElement('div');
            bookElement.classList.add('book-item');

            // SVG icon for the book
            const imgElement = document.createElement('img');
            const svgString = '<svg width="60" height="90" viewBox="0 0 24 36" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 32V4C4 2.89543 4.89543 2 6 2H20V34H6C4.89543 34 4 33.1046 4 32Z" fill="#e0e7ff"/><path d="M20 2H6C4.89543 2 4 2.89543 4 4V32C4 33.1046 4.89543 34 6 34H20V2Z" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16 2V34" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
            imgElement.src = `data:image/svg+xml,${encodeURIComponent(svgString)}`;
            imgElement.alt = 'Book Icon';
            
            const bookDetails = document.createElement('div');
            bookDetails.classList.add('book-details');

            const title = document.createElement('h3');
            title.textContent = book.book.title;

            const author = document.createElement('p');
            author.textContent = `Autor: ${book.book.author}`;

            const description = document.createElement('p');
            description.classList.add('book-description');
            description.textContent = book.book.description;

            const embedding = document.createElement('p');
            embedding.classList.add('book-embedding');
            const embeddingPreview = book.book.embedding.slice(0, 4).join(', ') + ', ...';
            embedding.innerHTML = `<strong>Embedding:</strong> ${embeddingPreview}`;

            const similarity = document.createElement('p');
            similarity.classList.add('book-similarity');
            similarity.innerHTML = `<strong>Similitud:</strong> ${Math.round(book.score * 100)}%`;

            bookDetails.appendChild(title);
            bookDetails.appendChild(author);
            bookDetails.appendChild(description);
            bookDetails.appendChild(embedding);
            bookDetails.appendChild(similarity);
            
            bookElement.appendChild(imgElement);
            bookElement.appendChild(bookDetails);

            resultsContainer.appendChild(bookElement);
        });
    }
});
