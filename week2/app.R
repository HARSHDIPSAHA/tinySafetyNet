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
library(scales)
library(stringr)

# Increase max upload size to 30MB
options(shiny.maxRequestSize = 30 * 1024^2)

# ================= UI =================
ui <- fluidPage(
  theme = bs_theme(version = 5, primary = "#007bff", base_font = font_google("Inter")),
  
  # --- Wavesurfer JS (For Audio Player) ---
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
      #waveform { border: 1px solid #444; border-radius: 5px; background: #222; margin-bottom: 10px; min-height: 150px; }
      .btn-audio { width: 100px; margin-right: 5px; }
      .shiny-options-group { color: inherit; }
      .alert-info { background-color: #2c3e50; color: #ecf0f1; border: 1px solid #34495e; }
      .card-header { font-weight: bold; font-size: 1.1em; }
    "))
  ),
  
  titlePanel("Audio Inference & Signal Analysis Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      input_dark_mode(id = "dark_mode", mode = "dark"),
      
      h4("1. Inference Data"),
      fileInput("files", "Upload Logs (CSV / Excel)", multiple = TRUE, accept = c(".csv", ".xlsx")),
      
      div(style = "padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin-bottom: 10px;",
          h6("Data Debugger:", style="margin-top:0; color: #00c3ff;"),
          textOutput("data_status_text")
      ),
      
      uiOutput("id_selector"),
      hr(),
      
      h4("2. Single Audio Analysis"),
      fileInput("audio_file", "Upload Audio (.wav)", multiple = FALSE, accept = c(".wav")),
      
      h5("Voice Changer"),
      radioButtons("voice_gender", "Target Voice:", choices = c("Female", "Male"), selected = "Female", inline = TRUE),
      actionButton("apply_voice", "Transform Audio", icon = icon("magic"), class = "btn-warning"),
      hr(),
      
      h5("Settings & Mapping"),
      p("Check boxes to map input labels (e.g., 'Fear') to groups (e.g., 'Unsafe').", style="font-size: 0.8em; opacity: 0.8;"),
      uiOutput("class_count_ui"),
      uiOutput("class_names_ui"),
      uiOutput("mapping_ui"),
      hr(),
      
      checkboxInput("show_all_history", "Show Full History (Ignore Date)", value = TRUE),
      conditionalPanel(
        condition = "input.show_all_history == false",
        sliderInput("days", "Filter History (Days):", min = 1, max = 365, value = 30)
      )
    ),
    
    mainPanel(
      tabsetPanel(
        # --- TAB 1: CLASS DISTRIBUTION ---
        tabPanel("Class Distribution", 
                 br(),
                 div(class="alert alert-info", 
                     strong("📊 Understanding Class Distribution:"), br(),
                     "This chart serves as a summary of the 'Danger Level'.", br(),
                     "- The height of each bar represents the total number of times that specific category (e.g., Safe vs. Unsafe) was detected.", br(),
                     "- If the 'Unsafe' bar is significantly high, it indicates a high-risk environment or frequent false positives."
                 ),
                 plotOutput("class_plot")),
        
        # --- TAB 2: TIMELINE ---
        tabPanel("Timeline", 
                 br(),
                 div(class="alert alert-info", 
                     strong("⏱️ Understanding Timeline:"), br(),
                     "This graph shows WHEN events occurred over time.", br(),
                     "- Spikes in the graph indicate bursts of activity or potential attacks.", br(),
                     "- Gaps indicate periods of silence or system inactivity."
                 ),
                 plotOutput("timeline_plot")),
        
        # --- TAB 3: EMOTION TRANSITIONS ---
        tabPanel("Emotion Transitions", 
                 br(),
                 div(class="alert alert-info", 
                     strong("🔄 Understanding Transition Matrix:"), br(),
                     "This heatmap visualizes 'Emotional Momentum'. It answers: 'If the user is currently Angry, what are they likely to become next?'", br(),
                     "- X-Axis: The NEXT emotion (Target).", br(),
                     "- Y-Axis: The PREVIOUS emotion (Source).", br(),
                     "- Diagonal Line: Stability. (e.g., Fear -> Fear means the person stayed afraid).", br(),
                     "- Off-Diagonal: Change. (e.g., Neutral -> Fear indicates the exact moment an attack might have started)."
                 ),
                 plotOutput("transition_plot", height = "500px")),
        
        # --- TAB 4: DAILY TREND ---
        tabPanel("Daily Trend", 
                 br(),
                 div(class="alert alert-info", 
                     strong("📅 Understanding Daily Trend:"), br(),
                     "This line graph aggregates data by day to show long-term patterns.", br(),
                     "- Useful for spotting specific days with unusual activity (e.g., 'Why were there 50 unsafe alerts on Tuesday?').", br(),
                     "- Each colored line represents a different emotion class."
                 ),
                 plotOutput("daily_plot")),
        
        # --- TAB 5: HEATMAP ---
        tabPanel("Hour-wise Heatmap", 
                 br(),
                 div(class="alert alert-info", 
                     strong("🔥 Understanding Hour-wise Heatmap:"), br(),
                     "This grid identifies the 'High Risk Hours' of the day.", br(),
                     "- Rows = Date, Columns = Hour of Day (0-23).", br(),
                     "- Lighter/Brighter colors indicate high activity.", br(),
                     "- Use this to identify patterns like 'Alerts always happen between 10 PM and 2 AM'."
                 ),
                 plotOutput("heatmap_plot")),
        
        # --- TAB 6: PER-ID ANALYSIS ---
        tabPanel("Per-ID Analysis", 
                 br(),
                 div(class="alert alert-info", 
                     strong("🆔 Understanding Per-ID Analysis:"), br(),
                     "This section isolates data for a single specific device or user.", br(),
                     "- Use the dropdown in the sidebar to switch between IDs.", br(),
                     "- The table provides exact percentages (e.g., 'Device 1 was 90% Safe, 10% Unsafe')."
                 ),
                 fluidRow(column(6, plotOutput("id_plot")), column(6, tableOutput("id_table")))),
        
        # --- TAB 7: SIGNAL ANALYSIS ---
        tabPanel("Signal Analysis",
                 br(),
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
                          div(class="alert alert-secondary", "📈 Pitch (Blue) shows vocal frequency. Energy (Orange) shows loudness. Panic usually causes both to spike."),
                          plotOutput("prosody_plot", height = "300px"),
                          br(),
                          h5("Spectrogram (Frequency Content)"),
                          plotOutput("spectrogram_plot", height = "300px"),
                          br(),
                          h5("Voice Transformation Result"),
                          uiOutput("modified_player_ui")
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
  
  observe({
    req(input$dark_mode)
    session$setCurrentTheme(bs_theme(version = 5, bootswatch = if(input$dark_mode == "dark") "darkly" else "flatly"))
  })
  
  # --- DATA LOADING ---
  raw_data <- reactive({
    req(input$files)
    files <- input$files
    data_list <- lapply(seq_len(nrow(files)), function(i) {
      ext <- tools::file_ext(files$name[i])
      
      # Try reading file
      df <- tryCatch({
        if (ext == "csv") read_csv(files$datapath[i], show_col_types = FALSE) 
        else read_excel(files$datapath[i])
      }, error = function(e) return(NULL))
      
      if(is.null(df)) return(NULL)
      
      colnames(df) <- tolower(colnames(df))
      
      # Renaming columns to standard format
      if ("inference_of_emotion" %in% colnames(df)) df <- df %>% rename(inference = inference_of_emotion)
      if ("time_log" %in% colnames(df)) df <- df %>% rename(time = time_log)
      
      # Standardize Text (Title Case)
      if ("inference" %in% colnames(df)) df$inference <- str_to_title(as.character(df$inference))
      
      # --- TIMESTAMP FIX (Handles HH:MM:SS format) ---
      if ("timestamp" %in% colnames(df)) {
        df$timestamp <- as.POSIXct(df$timestamp)
      } else if ("time" %in% colnames(df)) {
        df$time <- as.character(df$time)
        # Parse Assuming Today's Date if only Time is provided
        df$timestamp <- parse_date_time(df$time, orders = c("ymd HMS", "mdy HMS", "dmy HMS", "HMS"))
        
        if(any(is.na(df$timestamp))) {
          # Force Date if NA (happens when only HH:MM:SS is provided)
          df$timestamp[is.na(df$timestamp)] <- as.POSIXct(paste(Sys.Date(), df$time[is.na(df$timestamp)]), format="%Y-%m-%d %H:%M:%S")
        }
      } else {
        # Last resort: Create dummy sequence
        df$timestamp <- seq(from = Sys.time(), by = "-1 min", length.out = nrow(df))
      }
      
      # --- ID FIX (Detect if ID is just a row counter) ---
      if (!"id" %in% colnames(df)) {
        df$id <- "Device_1"
      } else {
        # Check if ID looks like 1, 2, 3... (Row Index)
        unique_ids <- n_distinct(df$id)
        if (unique_ids > (nrow(df) * 0.9)) { 
          # If 90% of rows have unique IDs, it's likely a counter, not a Device ID.
          # Overwrite it to treat as single stream for Transitions
          df$id_original <- df$id
          df$id <- "Stream_1" 
        }
      }
      
      if (!"inference" %in% colnames(df)) df$inference <- "Unknown"
      
      df %>% select(id, timestamp, inference)
    })
    
    bind_rows(data_list)
  })
  
  output$id_selector <- renderUI({ req(raw_data()); selectInput("device_id", "Select Device ID:", choices = unique(raw_data()$id)) })
  output$class_count_ui <- renderUI({ numericInput("num_classes", "Classes:", value = 2, min = 1) })
  
  output$class_names_ui <- renderUI({ 
    req(input$num_classes)
    lapply(1:input$num_classes, function(i) {
      def <- if(i==1) "Safe" else "Unsafe"
      textInput(paste0("class_", i), paste("Label", i), value = def)
    }) 
  })
  
  # Auto-Mapping Logic
  output$mapping_ui <- renderUI({ 
    req(input$num_classes, raw_data())
    emotions <- unique(raw_data()$inference)
    lapply(1:input$num_classes, function(i) {
      lbl <- input[[paste0("class_", i)]]
      sel <- c()
      if (lbl == "Safe") sel <- emotions[emotions %in% c("Neutral", "Happy", "Calm", "Surprise")]
      if (lbl == "Unsafe") sel <- emotions[emotions %in% c("Fear", "Angry", "Sad", "Disgust")]
      checkboxGroupInput(paste0("map_", i), paste("Map to", lbl), choices = emotions, selected = sel, inline = TRUE)
    }) 
  })
  
  classified_data <- reactive({
    req(raw_data(), input$num_classes)
    df <- raw_data(); df$class <- "Unassigned"
    for(i in 1:input$num_classes) {
      mapped <- input[[paste0("map_", i)]]; lbl <- input[[paste0("class_", i)]]
      if(!is.null(mapped)) df$class[df$inference %in% mapped] <- lbl
    }
    if (!input$show_all_history) df <- df %>% filter(timestamp >= (Sys.time() - days(input$days)))
    df
  })
  
  output$data_status_text <- renderText({
    req(classified_data())
    paste("Rows:", nrow(classified_data()), "| Mapped:", sum(classified_data()$class != "Unassigned"), "| Range:", min(classified_data()$timestamp), "to", max(classified_data()$timestamp))
  })
  
  # --- THEME FIX: Forces Dark Background on Plot ---
  get_theme <- function() {
    is_dark <- !is.null(input$dark_mode) && input$dark_mode == "dark"
    bg_col <- if(is_dark) "#2b2b2b" else "white"
    txt_col <- if(is_dark) "#ffffff" else "#000000"
    
    list(
      theme_minimal(base_size = 14),
      theme(
        plot.background = element_rect(fill = bg_col, color = NA),
        panel.background = element_rect(fill = bg_col, color = NA),
        text = element_text(color = txt_col),
        axis.text = element_text(color = txt_col),
        title = element_text(color = txt_col, face="bold"),
        legend.text = element_text(color = txt_col),
        legend.background = element_rect(fill = bg_col, color = NA),
        legend.position = "bottom"
      )
    )
  }
  
  output$transition_plot <- renderPlot({
    req(classified_data())
    trans_data <- classified_data() %>%
      arrange(id, timestamp) %>%
      group_by(id) %>%
      mutate(prev_class = lag(class)) %>%
      filter(!is.na(prev_class)) %>%
      ungroup() %>%
      count(prev_class, class) %>%
      group_by(prev_class) %>%
      mutate(prob = n / sum(n)) %>%
      ungroup()
    
    if (nrow(trans_data) == 0) return(ggplot() + annotate("text", x=1, y=1, label="No sequential data found.\n(Check mapping or data format)", color="grey", size=5) + theme_void())
    
    ggplot(trans_data, aes(x = class, y = prev_class, fill = prob)) +
      geom_tile(color = "white") +
      geom_text(aes(label = percent(prob, accuracy = 1)), color = "white", size = 6, fontface = "bold") +
      scale_fill_viridis_c(option = "magma", labels = percent) +
      labs(title = "Emotion Transition Matrix", x = "Target State", y = "Source State", fill = "Probability") +
      get_theme()
  })
  
  output$class_plot <- renderPlot({ req(classified_data()); ggplot(classified_data(), aes(class, fill=class)) + geom_bar() + scale_fill_viridis_d(option="turbo") + labs(title="Event Count by Class") + get_theme() })
  output$timeline_plot <- renderPlot({ req(classified_data()); ggplot(classified_data(), aes(timestamp, fill=class)) + geom_histogram(bins=50) + scale_fill_viridis_d(option="turbo") + labs(title="Event Timeline") + get_theme() })
  output$daily_plot <- renderPlot({ req(classified_data()); classified_data() %>% mutate(d=as.Date(timestamp)) %>% count(d, class) %>% ggplot(aes(d, n, color=class)) + geom_line(size=1.2) + labs(title="Daily Trends") + get_theme() })
  output$heatmap_plot <- renderPlot({ req(classified_data()); classified_data() %>% mutate(h=hour(timestamp), d=as.Date(timestamp)) %>% count(d, h) %>% ggplot(aes(h, d, fill=n)) + geom_tile() + labs(title="Activity Heatmap") + get_theme() })
  output$id_plot <- renderPlot({ req(input$device_id); classified_data() %>% filter(id==input$device_id) %>% ggplot(aes(class, fill=class)) + geom_bar() + labs(title=paste("Data for:", input$device_id)) + get_theme() })
  output$id_table <- renderTable({ req(input$device_id); classified_data() %>% filter(id==input$device_id) %>% count(class) %>% mutate(Pct=n/sum(n)*100) })
  
  # ================= AUDIO ANALYSIS =================
  audio_obj <- reactive({ req(input$audio_file); tryCatch({ tuneR::readWave(input$audio_file$datapath) }, error = function(e) NULL) })
  observeEvent(input$audio_file, { req(input$audio_file); session$sendCustomMessage("load_audio_js", list(data_uri = base64enc::dataURI(file = input$audio_file$datapath, mime = "audio/wav"))) })
  
  modified_audio <- reactiveVal(NULL)
  observeEvent(input$apply_voice, {
    req(audio_obj()); snd <- audio_obj(); if(snd@stereo) snd <- mono(snd, "both")
    factor <- if(input$voice_gender == "Male") 0.85 else 1.2
    snd_shifted <- snd; snd_shifted@samp.rate <- as.integer(snd@samp.rate * factor)
    tmp_file <- tempfile(fileext = ".wav"); tuneR::writeWave(snd_shifted, tmp_file)
    modified_audio(base64enc::dataURI(file = tmp_file, mime = "audio/wav"))
  })
  output$modified_player_ui <- renderUI({ req(modified_audio()); tagList(h5(paste("Converted to:", input$voice_gender), style="color: #00c3ff;"), tags$audio(src = modified_audio(), type = "audio/wav", controls = TRUE, style = "width: 100%;")) })
  
  output$prosody_plot <- renderPlot({
    req(audio_obj()); snd <- audio_obj(); if(snd@stereo) snd <- mono(snd, "both")
    if (length(snd@left) < 1000) return(NULL) 
    f0 <- tryCatch({ seewave::fund(snd, f=snd@samp.rate, ovlp=50, threshold=5, plot=FALSE, ylim=c(0, 0.4)) }, error = function(e) NULL)
    rms <- tryCatch({ seewave::rms(snd, f=snd@samp.rate, ovlp=50, plot=FALSE) }, error = function(e) NULL)
    df <- data.frame()
    if (!is.null(f0) && is.matrix(f0) && nrow(f0) > 0) { val <- !is.na(f0[,2]); if(sum(val)>0) df <- rbind(df, data.frame(Time=f0[val,1], Value=f0[val,2]*1000, Type="Pitch (Hz)")) }
    if (!is.null(rms) && is.matrix(rms) && nrow(rms) > 0) { val <- !is.na(rms[,2]); if(sum(val)>0) df <- rbind(df, data.frame(Time=rms[val,1], Value=rms[val,2]*500, Type="Energy (Scaled)")) }
    if (nrow(df) == 0) return(ggplot() + annotate("text", x=1, y=1, label="No Signal Detected", color="white") + theme_void())
    ggplot(df, aes(x=Time, y=Value, color=Type)) + geom_line(linewidth=1) + scale_color_manual(values=c("Pitch (Hz)"="#00c3ff", "Energy (Scaled)"="#ff7b00")) + get_theme()
  })
  
  output$spectrogram_plot <- renderPlot({ req(audio_obj()); snd <- audio_obj(); if(snd@stereo) snd <- mono(snd, "both"); par(bg=NA, col.lab="white", col.axis="white", fg="white"); if(!is.null(input$dark_mode) && input$dark_mode == "flatly") par(col.lab="black", col.axis="black", fg="black"); seewave::spectro(snd, f=snd@samp.rate, scale=FALSE, palette=temp.colors, grid=FALSE) })
  output$audio_features_table <- renderTable({
    req(audio_obj()); snd <- audio_obj(); if(snd@stereo) snd <- mono(snd, "both")
    spec <- tryCatch(seewave::specprop(seewave::spec(snd, f=snd@samp.rate, plot=FALSE)), error=function(e) list(mean=0, sh=0))
    fund <- tryCatch(seewave::fund(snd, f=snd@samp.rate, plot=FALSE, max=280), error=function(e) matrix(c(0,0), ncol=2)); avg <- if(is.matrix(fund)) mean(fund[,2], na.rm=TRUE)*1000 else 0
    data.frame(Feature = c("Duration", "Sample Rate", "Mean Freq", "Spectral Entropy", "Dominant Pitch"), Value = c(paste(round(length(snd@left)/snd@samp.rate, 2), "s"), paste(snd@samp.rate, "Hz"), paste(round(spec$mean/1000, 2), "kHz"), round(spec$sh, 3), paste(round(avg, 1), "Hz")))
  })
}

shinyApp(ui, server)
