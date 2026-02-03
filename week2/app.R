library(shiny)
library(readxl)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
library(bslib)
library(tuneR)
library(seewave)
library(jsonlite)
library(viridis)
library(base64enc)

# Increase max upload size to 30MB
options(shiny.maxRequestSize = 30 * 1024^2)

# ================= UI =================
ui <- fluidPage(
  theme = bs_theme(version = 5, primary = "#007bff", base_font = font_google("Inter")),
  
  # --- Wavesurfer JS ---
  tags$head(
    tags$script(src = "https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"),
    tags$script(HTML("
      var wavesurfer;
      $(document).on('shiny:connected', function() {
        wavesurfer = WaveSurfer.create({
          container: '#waveform',
          waveColor: '#00c3ff',
          progressColor: '#007bff',
          cursorColor: '#ffffff',
          barWidth: 2,
          barGap: 1,
          height: 150,
          responsive: true,
          normalize: true
        });
        Shiny.addCustomMessageHandler('load_audio_js', function(message) {
          wavesurfer.load(message.data_uri);
          $('#play_btn').html('<i class=\"fa fa-play\"></i> Play'); 
        });
        $('#play_btn').click(function() {
          wavesurfer.playPause();
          var icon = wavesurfer.isPlaying() ? '<i class=\"fa fa-pause\"></i> Pause' : '<i class=\"fa fa-play\"></i> Play';
          $(this).html(icon);
        });
        $('#stop_btn').click(function() {
          wavesurfer.stop();
          $('#play_btn').html('<i class=\"fa fa-play\"></i> Play');
        });
      });
    ")),
    tags$style(HTML("
      #waveform { border: 1px solid #444; border-radius: 5px; background: #222; margin-bottom: 10px; }
      .btn-audio { width: 100px; margin-right: 5px; }
    "))
  ),
  
  titlePanel("Audio Inference & Signal Analysis Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      input_dark_mode(id = "dark_mode", mode = "dark"),
      
      h4("1. Inference Data"),
      fileInput("files", "Upload Logs (CSV / Excel)", multiple = TRUE, accept = c(".csv", ".xlsx")),
      uiOutput("id_selector"),
      hr(),
      
      h4("2. Single Audio Analysis"),
      fileInput("audio_file", "Upload Audio (.wav)", multiple = FALSE, accept = c(".wav")),
      
      h5("Voice Changer"),
      radioButtons("voice_gender", "Target Voice:", choices = c("Female", "Male"), selected = "Female", inline = TRUE),
      actionButton("apply_voice", "Transform Audio", icon = icon("magic"), class = "btn-warning"),
      hr(),
      
      h5("Settings"),
      uiOutput("class_count_ui"),
      uiOutput("class_names_ui"),
      uiOutput("mapping_ui"),
      hr(),
      sliderInput("days", "Filter History (Days):", min = 1, max = 30, value = 7)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Class Distribution", plotOutput("class_plot")),
        tabPanel("Timeline", plotOutput("timeline_plot")),
        tabPanel("Daily Trend", plotOutput("daily_plot")),
        tabPanel("Hour-wise Heatmap", plotOutput("heatmap_plot")),
        tabPanel("Per-ID Analysis", fluidRow(column(6, plotOutput("id_plot")), column(6, tableOutput("id_table")))),
        
        tabPanel("Signal Analysis",
                 br(),
                 # --- Waveform Player ---
                 div(class = "card p-3",
                     h4("Interactive Waveform Player"),
                     div(id = "waveform"), 
                     div(
                       actionButton("play_btn", "Play", icon = icon("play"), class = "btn-primary btn-audio"),
                       actionButton("stop_btn", "Stop", icon = icon("stop"), class = "btn-danger btn-audio")
                     )
                 ),
                 br(),
                 fluidRow(
                   column(8,
                          h5("Prosody Analysis (Pitch & Energy)"),
                          p("Emotion is heavily conveyed through 'Prosody' (changes in pitch and loudness)."),
                          plotOutput("prosody_plot", height = "300px"),
                          br(),
                          h5("Spectrogram (Frequency Content)"),
                          plotOutput("spectrogram_plot", height = "300px"),
                          br(),
                          h5("Voice Transformation Result"),
                          uiOutput("modified_player_ui") # Modified player appears here
                   ),
                   column(4,
                          div(class = "card p-3",
                              h5("Acoustic Features"),
                              tableOutput("audio_features_table"),
                              hr(),
                              h5("Feature Guide"),
                              helpText(tags$ul(
                                tags$li(strong("Pitch (F0):"), "High variations often indicate excitement or anger."),
                                tags$li(strong("Spectral Entropy:"), "Complexity of sound. 0=Pure Tone, 1=Noise."),
                                tags$li(strong("Energy (RMS):"), "Loudness intensity over time.")
                              ))
                          )
                   )
                 )
        )
      )
    )
  )
)

# ================= SERVER =================
server <- function(input, output, session) {
  
  # 1. Theme
  observe({
    req(input$dark_mode)
    session$setCurrentTheme(bs_theme(version = 5, bootswatch = if(input$dark_mode == "dark") "darkly" else "flatly"))
  })
  
  # 2. Data Loading
  raw_data <- reactive({
    req(input$files)
    files <- input$files
    
    data_list <- lapply(seq_len(nrow(files)), function(i) {
      ext <- tools::file_ext(files$name[i])
      
      df <- if (ext == "csv") {
        readr::read_csv(files$datapath[i], show_col_types = FALSE)
      } else {
        readxl::read_excel(files$datapath[i])
      }
      
      colnames(df) <- tolower(colnames(df))
      
      # ---- ensure columns exist (match old behavior) ----
      if (!all(c("id", "time", "inference") %in% colnames(df)) &&
          ncol(df) >= 3) {
        colnames(df)[1:3] <- c("id", "time", "inference")
      }
      
      # ---- timestamp creation (same as old, but safer) ----
      if ("timestamp" %in% colnames(df)) {
        df$timestamp <- as.POSIXct(df$timestamp)
      } else if ("time" %in% colnames(df)) {
        df$time <- as.character(df$time)
        df$timestamp <- as.POSIXct(
          paste(Sys.Date(), df$time),
          format = "%Y-%m-%d %H:%M:%S"
        )
      } else {
        stop("Missing 'time' or 'timestamp' column")
      }
      
      if (!"id" %in% colnames(df)) df$id <- "Unknown"
      
      df %>% dplyr::select(id, timestamp, inference)
    })
    
    dplyr::bind_rows(data_list)
  })
  
  # 3. Dynamic Inputs
  output$id_selector <- renderUI({ req(raw_data()); selectInput("device_id", "Select Device ID:", choices = unique(raw_data()$id)) })
  output$class_count_ui <- renderUI({ numericInput("num_classes", "Classes:", value = 2, min = 1) })
  output$class_names_ui <- renderUI({ req(input$num_classes); lapply(1:input$num_classes, function(i) textInput(paste0("class_", i), paste("Label", i), if(i==1)"Safe" else "Unsafe")) })
  output$mapping_ui <- renderUI({ req(input$num_classes, raw_data()); lapply(1:input$num_classes, function(i) checkboxGroupInput(paste0("map_", i), paste("Map to", input[[paste0("class_", i)]]), unique(raw_data()$inference), inline=TRUE)) })
  
  # 4. Classification Logic
  classified_data <- reactive({
    req(raw_data(), input$num_classes)
    df <- raw_data(); df$class <- "Unassigned"
    for(i in 1:input$num_classes) {
      mapped <- input[[paste0("map_", i)]]; lbl <- input[[paste0("class_", i)]]
      if(!is.null(mapped)) df$class[df$inference %in% mapped] <- lbl
    }
    df %>% filter(timestamp >= (Sys.time() - days(input$days)))
  })
  
  # 5. Plots
  my_theme <- function() {
    list(
      theme_minimal(base_size = 14),
      theme(
        plot.background  = element_rect(fill = "transparent", color = NA),
        panel.background = element_rect(fill = "transparent", color = NA),
        legend.position  = "bottom"
      )
    )
  }
  output$class_plot <- renderPlot({ req(classified_data()); ggplot(classified_data(), aes(class, fill=class)) + geom_bar() + scale_fill_viridis_d(option="turbo") + my_theme() })
  output$timeline_plot <- renderPlot({ req(classified_data()); ggplot(classified_data(), aes(timestamp, fill=class)) + geom_histogram(bins=50) + scale_fill_viridis_d(option="turbo") + my_theme() })
  output$daily_plot <- renderPlot({ req(classified_data()); classified_data() %>% mutate(d=as.Date(timestamp)) %>% count(d, class) %>% ggplot(aes(d, n, color=class)) + geom_line() + my_theme() })
  output$heatmap_plot <- renderPlot({ req(classified_data()); classified_data() %>% mutate(h=hour(timestamp), d=as.Date(timestamp)) %>% count(d, h) %>% ggplot(aes(h, d, fill=n)) + geom_tile() + my_theme() })
  output$id_plot <- renderPlot({ req(input$device_id); classified_data() %>% filter(id==input$device_id) %>% ggplot(aes(class, fill=class)) + geom_bar() + my_theme() })
  output$id_table <- renderTable({ req(input$device_id); classified_data() %>% filter(id==input$device_id) %>% count(class) %>% mutate(Pct=n/sum(n)*100) })
  
  # ================= AUDIO ANALYSIS =================
  
  # A. Audio Object Reader
  audio_obj <- reactive({
    req(input$audio_file)
    tryCatch({ tuneR::readWave(input$audio_file$datapath) }, error = function(e) NULL)
  })
  
  # B. Send to JS Player
  observeEvent(input$audio_file, {
    req(input$audio_file)
    b64 <- dataURI(file = input$audio_file$datapath, mime = "audio/wav")
    session$sendCustomMessage("load_audio_js", list(data_uri = b64))
  })
  
  # C. Voice Changer Logic (SEPARATE FROM PLOTS)
  modified_audio <- reactiveVal(NULL)
  
  observeEvent(input$apply_voice, {
    req(audio_obj())
    snd <- audio_obj()
    if(snd@stereo) snd <- mono(snd, "both")
    
    # Pitch shift logic
    factor <- if(input$voice_gender == "Male") 0.85 else 1.2
    new_rate <- snd@samp.rate * factor
    snd_shifted <- snd
    snd_shifted@samp.rate <- as.integer(new_rate)
    
    tmp_file <- tempfile(fileext = ".wav")
    tuneR::writeWave(snd_shifted, tmp_file)
    
    b64_mod <- base64enc::dataURI(file = tmp_file, mime = "audio/wav")
    modified_audio(b64_mod)
  })
  
  output$modified_player_ui <- renderUI({
    req(modified_audio())
    tagList(
      h5(paste("Converted to:", input$voice_gender), style="color: #00c3ff;"),
      tags$audio(src = modified_audio(), type = "audio/wav", controls = TRUE, style = "width: 100%;")
    )
  })
  
  # D. Prosody Plot (Pitch & Energy)
  output$prosody_plot <- renderPlot({
    req(audio_obj())
    snd <- audio_obj()
    if(snd@stereo) snd <- mono(snd, "both")
    
    # Safe calculation
    f0_track <- tryCatch({ seewave::fund(snd, f=snd@samp.rate, ovlp=50, threshold=5, plot=FALSE, ylim=c(0, 0.4)) }, error = function(e) NULL)
    rms_track <- tryCatch({ seewave::rms(snd, f=snd@samp.rate, ovlp=50, plot=FALSE) }, error = function(e) NULL)
    
    df_pitch <- data.frame()
    if (!is.null(f0_track) && is.matrix(f0_track)) {
      valid <- !is.na(f0_track[,2])
      if(sum(valid) > 0) df_pitch <- data.frame(Time=f0_track[valid,1], Value=f0_track[valid,2]*1000, Type="Pitch (Hz)")
    }
    
    df_rms <- data.frame()
    if (!is.null(rms_track) && is.matrix(rms_track)) {
      valid <- !is.na(rms_track[,2])
      if(sum(valid) > 0) df_rms <- data.frame(Time=rms_track[valid,1], Value=rms_track[valid,2]*500, Type="Energy (Scaled)")
    }
    
    df_plot <- rbind(df_pitch, df_rms)
    
    if (nrow(df_plot) == 0) return(NULL)
    
    ggplot(df_plot, aes(x=Time, y=Value, color=Type)) +
      geom_line(linewidth=1) +
      scale_color_manual(values=c("Pitch (Hz)"="#00c3ff", "Energy (Scaled)"="#ff7b00")) +
      labs(title="Prosodic Features", x="Time (s)", y="Value") +
      my_theme()
  })
  
  # E. Spectrogram
  output$spectrogram_plot <- renderPlot({
    req(audio_obj())
    snd <- audio_obj()
    if(snd@stereo) snd <- mono(snd, "both")
    
    par(bg=NA, col.lab="white", col.axis="white", fg="white")
    if(input$dark_mode == "flatly") par(col.lab="black", col.axis="black", fg="black")
    seewave::spectro(snd, f=snd@samp.rate, scale=FALSE, palette=temp.colors, grid=FALSE)
  })
  
  # F. Features Table
  output$audio_features_table <- renderTable({
    req(audio_obj())
    snd <- audio_obj()
    if(snd@stereo) snd <- mono(snd, "both")
    
    spec <- tryCatch(seewave::specprop(seewave::spec(snd, f=snd@samp.rate, plot=FALSE)), error=function(e) list(mean=0, sh=0))
    fund <- tryCatch(seewave::fund(snd, f=snd@samp.rate, plot=FALSE, max=280), error=function(e) matrix(c(0,0), ncol=2))
    avg_pitch <- mean(fund[,2], na.rm=TRUE) * 1000
    
    data.frame(
      Feature = c("Duration", "Sample Rate", "Mean Freq", "Spectral Entropy", "Dominant Pitch"),
      Value = c(paste(round(length(snd@left)/snd@samp.rate, 2), "s"), paste(snd@samp.rate, "Hz"),
                paste(round(spec$mean/1000, 2), "kHz"), round(spec$sh, 3), paste(round(avg_pitch, 1), "Hz"))
    )
  })
}

shinyApp(ui, server)
