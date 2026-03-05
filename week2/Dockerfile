# Use the official R Shiny image
FROM rocker/shiny:4.3.2

# Install Linux system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all required R packages using the fast Posit Package Manager
RUN R -e "options(repos = c(RSPM = 'https://packagemanager.posit.co/cran/__linux__/jammy/latest')); \
    install.packages(c('shiny', 'readxl', 'readr', 'dplyr', 'ggplot2', 'lubridate', 'tidyr', 'bslib', 'tuneR', 'seewave', 'jsonlite', 'viridis', 'base64enc', 'scales', 'stringr'))"

# Copy your Shiny app into the container
WORKDIR /srv/shiny-server/
COPY app.R /srv/shiny-server/app.R

# Run the app on port 3838
EXPOSE 3838
CMD ["R", "-e", "shiny::runApp('/srv/shiny-server/app.R', host = '0.0.0.0', port = 3838)"]