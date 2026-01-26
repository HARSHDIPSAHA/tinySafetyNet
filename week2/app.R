library(shiny)
library(readxl)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
library(bslib)

# ================= UI =================
ui <- fluidPage(
  
  theme = bs_theme(
    version = 5
  ),
  
  titlePanel("Audio Inference Research Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      input_dark_mode(id = "dark_mode", mode = "dark"),
      
      fileInput(
        "files",
        "Upload Dataset(s) (CSV / Excel)",
        multiple = TRUE,
        accept = c(".csv", ".xlsx")
      ),
      
      uiOutput("id_selector"),
      uiOutput("class_count_ui"),
      uiOutput("class_names_ui"),
      uiOutput("mapping_ui"),
      
      sliderInput(
        "days",
        "Show last N days:",
        min = 1, max = 10, value = 1
      )
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Class Distribution", plotOutput("class_plot"), helpText(
          "This bar chart shows the total number of audio samples belonging to each "
          ,"high-level class after emotion-to-class mapping. "
          ,"It helps identify class imbalance and dominant emotional patterns."
        )),
        tabPanel("Timeline", plotOutput("timeline_plot"), helpText(
          "This timeline visualizes emotion or class predictions across time. "
          ,"Each point represents an inference event, enabling analysis of temporal "
          ,"emotion transitions."
        )),
        tabPanel("Daily Trend", plotOutput("daily_plot"), helpText(
          "This plot aggregates emotion or class occurrences on a daily basis. "
          ,"It reveals long-term trends and day-wise variations in emotional states."
        )),
        tabPanel("Hour-wise Heatmap", plotOutput("heatmap_plot"), helpText(
          "This heatmap displays the distribution of emotions or classes across "
          ,"different hours of the day. Darker cells indicate higher frequency, "
          ,"highlighting peak activity periods."
        )),
        tabPanel("Per-ID Analysis",
                 plotOutput("id_plot"),
                 tableOutput("id_table"), helpText(
                   "This section provides a detailed analysis for a selected device or subject ID. "
                   ,"The plot and table together allow individual-level emotional behavior analysis."
                 ))
      )
    )
  )
)

# ================= SERVER =================
server <- function(input, output, session) {
  observe({
    req(input$dark_mode)
    
    session$setCurrentTheme(
      bs_theme(
        version = 5,
        bootswatch = if (input$dark_mode == "dark") "darkly" else "flatly"
      )
    )
  })
  
  # -------- Load Data --------
  raw_data <- reactive({
    req(input$files)
    
    files <- input$files
    
    data_list <- lapply(seq_len(nrow(files)), function(i) {
      ext <- tools::file_ext(files$name[i])
      
      df <- if (ext == "csv") {
        read_csv(files$datapath[i], show_col_types = FALSE)
      } else {
        read_excel(files$datapath[i])
      }
      
      colnames(df)[1:3] <- c("id", "time", "inference")
      
      df %>%
        mutate(
          time = as.character(time),
          timestamp = as.POSIXct(
            paste(Sys.Date(), time),
            format = "%Y-%m-%d %H:%M:%S"
          )
        )
    })
    
    bind_rows(data_list)
  })
  
  # -------- UI Controls --------
  output$id_selector <- renderUI({
    req(raw_data())
    selectInput("device_id", "Select Device ID:",
                choices = unique(raw_data()$id))
  })
  
  output$class_count_ui <- renderUI({
    numericInput("num_classes", "Number of Classes:", value = 2, min = 1)
  })
  
  output$class_names_ui <- renderUI({
    req(input$num_classes)
    lapply(seq_len(input$num_classes), function(i) {
      textInput(paste0("class_", i), paste("Class", i), paste("Class", i))
    })
  })
  
  emotions <- reactive({
    unique(raw_data()$inference)
  })
  
  output$mapping_ui <- renderUI({
    req(input$num_classes, emotions())
    lapply(seq_len(input$num_classes), function(i) {
      checkboxGroupInput(
        paste0("map_", i),
        paste("Emotions for", input[[paste0("class_", i)]]),
        emotions()
      )
    })
  })
  
  # -------- Classified Data --------
  classified_data <- reactive({
    req(raw_data(), input$num_classes)
    
    df <- raw_data()
    df$class <- "Unassigned"
    
    for (i in seq_len(input$num_classes)) {
      mapped <- input[[paste0("map_", i)]]
      if (!is.null(mapped)) {
        df$class[df$inference %in% mapped] <- input[[paste0("class_", i)]]
      }
    }
    
    cutoff <- Sys.time() - days(input$days)
    df %>% filter(timestamp >= cutoff)
  })
  
  # ================= PLOTS =================
  
  output$class_plot <- renderPlot({
    df <- classified_data()
    req(nrow(df) > 0)
    ggplot(df, aes(class, fill = class)) + geom_bar()
  })
  
  output$timeline_plot <- renderPlot({
    df <- classified_data()
    req(nrow(df) > 0)
    ggplot(df, aes(timestamp, fill = class)) +
      geom_histogram(binwidth = 300)
  })
  
  output$daily_plot <- renderPlot({
    classified_data() %>%
      mutate(date = as.Date(timestamp)) %>%
      count(date, class) %>%
      ggplot(aes(date, n, color = class)) +
      geom_line() +
      geom_point()
  })
  
  output$heatmap_plot <- renderPlot({
    classified_data() %>%
      mutate(hour = hour(timestamp),
             date = as.Date(timestamp)) %>%
      count(date, hour) %>%
      ggplot(aes(hour, date, fill = n)) +
      geom_tile()
  })
  
  output$id_plot <- renderPlot({
    classified_data() %>%
      filter(id == input$device_id) %>%
      ggplot(aes(class, fill = class)) +
      geom_bar()
  })
  
  output$id_table <- renderTable({
    classified_data() %>%
      filter(id == input$device_id) %>%
      count(class)
  })
}

shinyApp(ui, server)
