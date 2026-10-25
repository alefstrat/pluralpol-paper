#This function computes a mixing matrix from an igraph file. 
make_mixing_matrix <- function(fn, drop_pol_verts) {
  
  g <- readRDS(fn)
  
  # drop any political vertices
  if (drop_pol_verts == TRUE) {
    political_subs <- read.csv("subreddit_topic_hierarchy_final.csv") %>%
      filter(topic_level4 == "Politics and social issues") %>% 
      pull(subreddit)
    
    g <- delete_vertices(g, intersect(political_subs, V(g)$name))
  }
  
  
  edge_list <- igraph::as_data_frame(g) |>
    select(-starts_with("activity"))
  
  lean_lookup <- tibble(
    name  = V(g)$name,
    lean  = V(g)$lean_e,
    score = V(g)$score_e
  )
  
  mixing_matrix <- edge_list |>
    left_join(lean_lookup, by = c("from" = "name")) |>
    left_join(lean_lookup, by = c("to" = "name")) |>
    count(lean.x, lean.y) |>
    pivot_wider(names_from = lean.y, values_from = n) |>
    column_to_rownames("lean.x") |>
    #select(-c("lean.x")) |>
    as.matrix()
  
  
  return(mixing_matrix)
}

make_mixing_matrix_topic <- function(fn, tl, selected_topic, df) {
  
  g <- readRDS(fn)
  
  subs <- df %>%
    filter(.data[[tl]] == selected_topic) %>% 
    pull(subreddit)
  
  
  g <- induced_subgraph(g, vids=which(V(g)$name%in%subs))
  
  
  #V(g)$lean_e <- ifelse(V(g)$percent_score_rank_e>= 0.5000, "left", "right")
  
  edge_list <- igraph::as_data_frame(g) |>
    select(-starts_with("activity"))
  
  lean_lookup <- tibble(
    name  = V(g)$name,
    lean  = V(g)$lean_e,
    score = V(g)$score_e
  )
  
  categories <- c("left", "mixed", "right")
  
  mixing_matrix <- edge_list |>
    left_join(lean_lookup, by = c("from" = "name")) |>
    left_join(lean_lookup, by = c("to" = "name")) |>
    # Make both columns to contain both categories
    mutate(
      lean.x = factor(lean.x, levels = categories),
      lean.y = factor(lean.y, levels = categories)
    ) |>
    # .drop = FALSE to force count() to include 0 for empty combos
    count(lean.x, lean.y, .drop = FALSE) |>
    pivot_wider(names_from = lean.y, values_from = n, values_fill = 0) |>
    column_to_rownames("lean.x") |>
    as.matrix()
  
  
  return(mixing_matrix)
}

#This function simply creates a mixing matrix without the mixed category.
make_mixing_matrix_no_mixed <- function(fn, drop_pol_verts, drop_mixed = FALSE) {
  
  g <- readRDS(fn)
  
  if (drop_pol_verts==TRUE) {
    political_subs <- read.csv("subreddit_topic_hierarchy_final.csv") %>%
      filter(topic_level4 == "Politics and social issues") %>% 
      pull(subreddit)
    
    g <- delete_vertices(g, intersect(political_subs, V(g)$name))
  }
  
  if (drop_mixed==TRUE) {
    g <- delete_vertices(g, V(g)$lean_e == "mixed")
  }
  
  
  edge_list <- igraph::as_data_frame(g) |>
    select(-starts_with("activity"))
  
  V(g)$lean_e <- ifelse(V(g)$percent_score_rank_e>= 0.5000, "left", "right")
  
  lean_lookup <- tibble(
    name  = V(g)$name,
    lean  = V(g)$lean_e,
    score = V(g)$score_e
  )
  
  mixing_matrix <- edge_list |>
    left_join(lean_lookup, by = c("from" = "name")) |>
    left_join(lean_lookup, by = c("to" = "name")) |>
    count(lean.x, lean.y) |>
    pivot_wider(names_from = lean.y, values_from = n) |>
    column_to_rownames("lean.x") |>
    #select(-c("lean.x")) |>
    as.matrix()
  
  print(mixing_matrix)
  
  
  return(mixing_matrix)
}


# compute mixing matrix and return matrix of z-scores
zmm <- function(mmg){
  gmarg <- rowSums(abs(mmg))
  emmg <- (gmarg%o%gmarg)/sum(gmarg)
  print(sum(gmarg)==sum(abs(mmg)))
  dmmg <- abs(mmg)-emmg
  zmmg <- dmmg/sqrt(emmg)
  zmat <- data.frame(zmmg)
  return(zmat)
}

# Function to compute the observed mixing rate
mmr <- function(mmg){
  gmarg <- rowSums(abs(mmg))
  mmg_mr <- mmg/sum(gmarg)
  mrmat <- data.frame(mmg_mr)
  return(mrmat)
}

# Compute the expected mixing rate
mme <- function(mmg){
  gmarg <- rowSums(abs(mmg))
  emmg <- ((gmarg%o%gmarg)/sum(gmarg))/sum(gmarg)
  emmg <- data.frame(emmg)
  return(emmg)
}

# helper function
piv <- function(df){
  dfout <- df %>%
    rownames_to_column("from") %>% 
    pivot_longer(cols = !from, 
                 names_to = "to", 
                 values_to = "z-score")
  return(dfout)
}

# converting matrices of zcores to dataframes
convert_to_dataframe <- function(l){
  zmats <- lapply(l, zmm)
  pivoted_zmats <- lapply(zmats, piv)
  one_frame <- rbindlist(pivoted_zmats, idcol = TRUE)
  return(one_frame)
}

make_core_plot <- function(df){
  new_labels <- as_labeller(c(
    "left" = "Left Block", 
    "mixed" = "Mixed Block", 
    "right" = "Right Block"
  ))
  
  GROUP_COLORS <- c(left = "blue", 
                    mixed = "purple2", 
                    right = "red")
  
  p<-df %>%
    #filter(pol=="with_pol") %>%
    ggplot(aes(
      x = year, 
      y = `z-score`, 
      group = interaction(factor(ties), to) 
    )) +
    geom_line(aes(color = factor(ties)), linewidth = 0.6) +
    geom_point(aes(fill = to, color = after_scale(fill)), shape = 21, size = 1.4, alpha = 0.7) +
    scale_fill_manual(values = GROUP_COLORS, name = "Target Block", labels = c("Left", "Mixed", "Right")) +
    scale_color_grey(start = 0.9, end = 0.1, name = "Ties") +
    scale_x_continuous(breaks = seq(2008, 2025, by = 4)) +
    facet_wrap(~from, labeller = new_labels, axes = "all")
  return(p) 
}

make_plot_eo <- function(df, t, b){
  
  new_labels <- as_labeller(c(
    "left" = "Left", 
    "mixed" = "Mixed", 
    "right" = "Right"
  ))
  
  GROUP_COLORS  <- c(left = "blue", 
                     mixed = "purple2", 
                     right = "red")
  
  p <- df %>%
    ggplot(aes( x = year, y = `z-score`, group = interaction(factor(ties), to, type))) +
    geom_line(aes(color = to, linetype = type), linewidth = 0.4) +
    geom_point(
      data = ~ subset(., type == "Observed"),
      aes(fill = to, color = after_scale(fill)), 
      shape = 21, 
      size = 1.0, 
      alpha = 0.7) +
    scale_fill_manual(values = GROUP_COLORS, 
                      name = "Target", 
                      labels = c("Left", "Mixed", "Right")) +
    scale_color_manual(values = GROUP_COLORS, 
                       name = "Target", 
                       labels = c("Left", "Mixed", "Right")) +
    scale_linetype_manual(values = c("Observed" = "solid", "Expected" = "dotted"), 
                          name = "Metric") +
    scale_x_continuous(breaks = seq(2008, 2025, by = 4)) +
    facet_wrap(~from, labeller = new_labels, axes = "all") + 
    theme_classic(base_size = 7) + 
    theme(
      text = element_text(family = "sans"), 
      strip.background = element_blank(),       
      strip.text = element_text(face = "bold", size = 7), 
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5.5), 
      axis.text.y = element_text(size = 5.5),
      axis.title = element_text(size = 7),
      panel.grid.major.y = element_line(color = "gray95", linewidth = 0.3), 
      plot.title = element_text(margin = margin(t = t, b = b), face = "bold"),
      legend.position = "top",
      legend.title= element_text(size = 7, face = "bold"),
      legend.text= element_text(size = 7),
      legend.key.size= unit(0.25, "cm"),
      legend.spacing.x = unit(0.1, "cm")#,
      #legend.box.margin  = margin(t = -8) # change if need be
    ) + 
    ylab("Mixing Rate") + 
    xlab("Year") + 
    guides(fill = "legend", colour = "none", linetype = "legend") +
    ggtitle("Source:")
  return(p)
}

zscore_pipeline <- function(fold,
                            ties,
                            matrix_fn,
                            stat_fn = zmm,
                            ...,
                            start = 3,
                            base_year = 2007) {
  files <- list.files(fold, full.names = TRUE)[-seq_len(start - 1)]
  files %>%
    lapply(matrix_fn, ...) %>%
    lapply(stat_fn) %>%
    lapply(piv) %>%
    rbindlist(idcol = TRUE) %>%
    mutate(year = .id + base_year, ties = ties)
}

#wrapper functions so execution code does not blow up
mmm_wrapped <- function(fold, ties, no_pol, stat_fn){
  zscore_pipeline(fold, ties, make_mixing_matrix, stat_fn, drop_pol_verts = no_pol)}

mmm_wrapped_all_labelled <- function(fold, ties, no_pol){
  zscore_pipeline(fold, ties, make_mixing_matrix_no_mixed, zmm, drop_pol_verts = no_pol)}

mmm_wrapped_no_mixed <- function(fold, ties, no_pol){
  zscore_pipeline(fold, ties, make_mixing_matrix_no_mixed, zmm,
                  drop_pol_verts = no_pol, drop_mixed = TRUE)}

mmm_wrapped_topic <- function(fold, ties, tl, selected_topic, df){
  zscore_pipeline(fold, ties, make_mixing_matrix_topic, zmm,
                  tl = tl, selected_topic = selected_topic, df)
}

# generate zscores of mm across negative or positive net series and with a variety of tie configurations.
#Can include and exclude political subreddits via boolean of no_pol
# Change fp as needed to get to negative or positive networks
all_mm_results <- function(prefix, ties_low, ties_high, no_pol = FALSE){
  zn_mixed <- list()
  zn_mask_mixed <- list()
  zn_drop_mixed <- list()
  
  for (i in ties_low:ties_high){
    fp <- paste0("data/networks/",prefix, i, "_ties")
    zn_mixed[[i]] <- mmm_wrapped(fp, i, zmm, no_pol = FALSE)
    zn_mask_mixed[[i]] <- mmm_wrapped_all_labelled(fp, i, no_pol = FALSE)
    zn_drop_mixed[[i]] <- mmm_wrapped_no_mixed(fp, i, no_pol = FALSE)
  }
  
  zndf <- lapply(list(zn_mixed, zn_mask_mixed, zn_drop_mixed), function(x) do.call(rbind, x))
  return(zndf)
}


## Expected and observed functions and plots

eo_pipeline <- function(fold, 
                        ties, 
                        matrix_fn, 
                        ..., # pass in extra args if you need
                        start = 3, 
                        base_year = 2007) {
  files <- list.files(fold, full.names = TRUE)[-seq_len(start - 1)]
  mats  <- lapply(files, matrix_fn, ...)
  
  one_stat <- function(stat_fn, label) {
    mats %>%
      lapply(stat_fn) %>%
      lapply(piv) %>%
      rbindlist(idcol = TRUE) %>%
      mutate(year = .id + base_year, ties = ties, type = label)
  }
  
  rbind(one_stat(mmr, "Observed"), one_stat(mme, "Expected"))
}

# mixing rates for one network folder, all three variations
eo_variants <- function(fp, ties = 3, no_pol = FALSE) {
  list(
    mixed = eo_pipeline(fp, ties, make_mixing_matrix, drop_pol_verts = no_pol),
    mask  = eo_pipeline(fp, ties, make_mixing_matrix_no_mixed, drop_pol_verts = no_pol),
    drop  = eo_pipeline(fp, ties, make_mixing_matrix_no_mixed, drop_pol_verts = no_pol,
                        drop_mixed = TRUE)
  )
}

# make a grid plot here
eo_grid <- function(pos_df, neg_df) {
  p_pos <- make_plot_eo(pos_df, t = 18, b = -32) #changing element margins of ggtitle
  p_neg <- make_plot_eo(neg_df, t = 18, b = -32) + theme(legend.position = "none")
  
  plot_grid(p_pos, p_neg,
            labels = c("A: Positive", "B: Negative"),
            ncol = 1, align = "hv", axis = "lr",
            rel_heights = c(1, 1),
            label_size = 10, label_fontface = "bold")
}

