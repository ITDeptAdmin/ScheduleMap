# RAM Schedule Map System

This repository and plugin setup power the **RAM Schedule Map** used on the RAM website to display **upcoming clinics** in both map and calendar views.

This system connects:

- **Modern Events Calendar (MEC)** in WordPress
- the **MEC API / sync process**
- a **GitHub-hosted CSV**
- a **custom WordPress plugin**
- **Mapbox**
- **Google Analytics 4 (GA4)**
- **Looker Studio dashboards**

This document explains how the system works, how to maintain it, and what to check if something breaks.

---

# 1. What This System Does

The RAM Schedule Map displays **upcoming clinics** on the website using a custom interactive interface.

It allows visitors to:

- view clinics on a map
- switch to a calendar view
- filter by state
- filter by month
- filter by clinic type
- filter by services
- click clinics in the map or list
- click through to the clinic page

The map is powered by a CSV file stored in GitHub and loaded into a custom WordPress plugin.

---

# 2. How the Schedule Data Flow Works

The schedule system works in this order:

1. A clinic is created or updated in **Modern Events Calendar (MEC)** in WordPress.
2. The MEC data is pulled through the schedule sync process / MEC API connection.
3. GitHub Actions update the main CSV in this repository.
4. The WordPress schedule map plugin loads that CSV from GitHub.
5. The plugin parses the CSV in JavaScript and renders:
   - the map view
   - the clinic list
   - the calendar view
6. User interactions are tracked in **GA4**.
7. GA4 data is displayed in **Looker Studio dashboards**.

### Simple Architecture
```
WordPress (MEC)
↓
MEC API / sync
↓
GitHub CSV
↓
WordPress plugin
↓
Map + Calendar + List
↓
GA4 tracking
↓
Looker Studio reporting
```
---

# 3. GitHub Repository

## Repository name - ITDeptAdmin/ScheduleMap

This repository stores the schedule CSV used by the website.

### Important items in the repo

- `.github/workflows/`
- `scripts/`
- `Clinic Master Schedule for git.csv`
- `README.md`

### Main data file

The primary CSV used by the live schedule map is: Clinic Master Schedule for git.csv

This is the file the WordPress plugin loads from GitHub.

---

# 4. Where the Website Gets the Data

The schedule map plugin points to the **GitHub raw CSV URL** and loads it directly in the browser.

The plugin configuration includes:

- CSV URL
- Mapbox token
- Mapbox style
- sticky header selector

The CSV is fetched client-side, parsed with **Papa Parse**, then rendered into the map and calendar views.

---

# 5. Creating or Updating a Clinic

## Source of truth

The official schedule starts in **WordPress / Modern Events Calendar (MEC)**.

### General process

When adding or updating a clinic:

1. Go to the clinic/event in MEC.
2. Enter or update all required clinic information.
3. Publish or update the event.
4. The sync process updates the schedule CSV in GitHub.
5. The schedule map on the website reflects the updated CSV.

### Important event data

For the schedule map to work correctly, clinic records should include:

- title
- address
- city
- state
- latitude
- longitude
- start date
- end date
- clinic type / telehealth status
- services available
- event URL

If latitude/longitude or dates are missing, the clinic may not display correctly on the map.
* For latitude and longitude if you enter the address in MEC it should popup a google address you can click and it will take care of this for you.
* Make sure to add the tag of the state the event is in.  It would need to be spelled out ex. Tennessee
* For catagory make sure to select popup or telehealth clinic

---

# 6. WordPress Plugin

The custom plugin renders the **RAM Schedule Map** on the website.

## Main shortcode options

### Full-page version
[ram_schedule_map]

### Embedded version
[ram_schedule_map mode="embed" height="750px"]
*height number can be changed

## What the plugin does

The plugin:

- loads plugin settings
- fetches the CSV from GitHub
- parses clinic rows
- filters out canceled or expired clinics
- renders map markers
- renders the list view
- renders the calendar view
- tracks user interactions in GA4

### Display modes

The plugin supports two main display modes:

- **full** — full-page schedule map experience
- **embed** — embeddable version with configurable height

### Major front-end features

The schedule plugin includes:

- Mapbox map
- map/list/calendar switching
- state filter
- month filter
- clinic type filter
- service filter
- reset filters
- reset view
- back to top
- popup details
- learn more links
- calendar pill clicks

---

# 7. Map Display Logic

The schedule map:

- only shows clinics with valid coordinates
- hides canceled clinics
- hides clinics whose end date has already passed
- supports popup clinics and telehealth clinics
- colors telehealth and popup clinics differently
- automatically fits bounds to the filtered clinic set
- supports both map and calendar browsing

  *If a clinic has been posted and in MEC you make it a draft it will NOT automatacally remove it from the CSV file and the map.  However if you delete instead of making it a draft it will delete it.  If for somereason you need to leave it as draft you can remove it from the map manually by going to GitHub and editing the CSV file and deleteing it there.  Note this plugin and code does not auto add anything that is a event that is a draft.  This is just a random situation when it was already a posted event.

### Popup clinic rendering

When a user clicks a clinic marker, the plugin opens a popup containing:

- clinic title
- address
- clinic dates
- clinic time (if available)
- services
- telehealth indicator
- learn more button

### List rendering

The right-side list shows:

- city/state
- clinic title
- address
- dates
- services
- clinic type badge
- learn more link

### Calendar rendering

The calendar view shows:

- clinics placed on their matching dates
- telehealth vs popup visual styling
- clickable calendar pills that link to the clinic page

---

# 8. How the Plugin Tracks GA4

The schedule map sends custom events using `dataLayer.push()` first, with fallback to `gtag()` if needed.

All tracking includes base parameters such as:

- `ram_feature`
- `ram_map_version`
- `ram_dataset_version`
- `page_path`

### Main schedule GA4 events

#### Initialization / usage

- `ram_schedule_init`
- `ram_schedule_view_switch`

#### Filtering

- `ram_schedule_filter_change`
- `ram_schedule_reset_view`

#### Navigation / interaction

- `ram_schedule_card_click`
- `ram_schedule_map_dot_click`
- `ram_schedule_popup_open`
- `ram_schedule_learn_more_click`
- `ram_schedule_back_to_top`
- `ram_schedule_calendar_pill_click`

### Important tracked parameters

Depending on the event, the plugin can send:

- `instance_id`
- `filter_state`
- `filter_state_label`
- `filter_month`
- `filter_clinic_type`
- `filter_services`
- `reason`
- `service_key`
- `results_count`
- `place_key`
- `place_label`
- `place_city`
- `place_state`
- `place_country`
- `clinic_id`
- `clinic_title`
- `clinic_telehealth`
- `source`

### Why this tracking matters

This allows RAM to report on:

- how many users use the map
- which clinics get the most attention
- which states get the most interest
- which services users filter for
- whether users prefer popup or telehealth clinics
- whether users interact more with the list, map, or calendar

---

# 9. GA4 Custom Dimensions Used

The schedule map reporting depends on **event-scoped custom dimensions** in GA4.

Important schedule dimensions include:

- `Schedule Map_clinic_id`
- `Schedule Map_clinic_title`
- `Schedule Map_filter_clinic_type`
- `Schedule Map_filter_services`
- `Schedule Map_filter_state_label`
- `Schedule Map_month`
- `Schedule Map_reason`
- `Schedule Map_source`

These dimensions power the **Looker Studio dashboards** and **GA4 Explore reports**.

---

# 10. Looker Studio Dashboard

A Looker Studio dashboard has been created to visualize **Schedule Map usage**.

### Suggested dashboard sections

## Page 1 — Overview

Typical metrics shown:

- Total users
- Map Dot Clicks
- Clinic Clicks
- Learn More Clicks
- Top State
- Top Clinic
- Clinic Heat Map

## Page 2 — Behavior / filters

Typical charts shown:

- event name table
- services filtered
- state interest
- clinic engagement

### What the dashboard helps RAM answer

The dashboard helps show:

- how many people use the schedule map
- which states show the most clinic interest
- which clinics get the most engagement
- which services people are browsing
- how users move through the schedule tool

---

# 11. Best Practices for Reporting

When building reports, use event filters intentionally.

### Upcoming clinic engagement

Use events like:

- `ram_schedule_card_click`
- `ram_schedule_map_dot_click`
- `ram_schedule_learn_more_click`

### Filter usage

Use:

- `ram_schedule_filter_change`

Then segment by:

- `reason`
- `filter_services`
- `filter_state_label`
- `filter_clinic_type`

### Map usage funnel

Compare:

- `ram_schedule_init`
- `ram_schedule_filter_change`
- `ram_schedule_popup_open`
- `ram_schedule_card_click`
- `ram_schedule_learn_more_click`

---

# 12. Common Maintenance Tasks

### If a clinic is missing from the map

Check:

- Is the event published in MEC?
- Does it have valid latitude/longitude?
- Is the end date still in the future?
- Is it marked canceled?
- Did the GitHub CSV update?
- Does the clinic exist in `Clinic Master Schedule for git.csv`?

### If the map is blank

Check:

- Is the plugin active in WordPress?
- Is the GitHub CSV URL valid?
- Is the CSV reachable?
- Is the Mapbox token still valid?
- Is the Mapbox style still published?
- Are there console errors for Mapbox or Papa Parse?

### If analytics seem missing

Check:

- Is `dataLayer` present on the page?
- Is GTM firing the GA4 event tag?
- Does GA4 show the custom event in realtime or event reports?
- Are custom dimensions properly registered?
- Has enough processing time passed in GA4?

---

# 13. Important Notes About GA4 Limits

GA4 has a **hard limit on event-scoped custom dimensions**.

Because this property is heavily customized, be careful before adding new dimensions.

Before creating new custom dimensions:

- review whether an older one can be deleted
- avoid test-only dimensions
- prefer using existing fields when possible

---

# 14. Troubleshooting Checklist

### WordPress / MEC

- clinic exists
- clinic published
- required fields complete

### GitHub

- workflow ran successfully
- main CSV updated
- raw CSV URL works

### Plugin

- shortcode placed correctly
- plugin active
- JS/CSS loaded
- mapbox token/style valid

### Analytics

- GTM preview confirms event fires
- GA4 tag fires successfully
- custom definitions match event parameters
- Looker chart filters are correct

---

# 15. Shortcode Reference

### Full-page schedule map
[ram_schedule_map]

### Embedded schedule map
[ram_schedule_map mode="embed" height="750px"]

Use the embedded version when the map should appear inside another page layout without taking over the full page experience.

---

# 16. Summary

The RAM Schedule Map system is built so that clinic data:

1. starts in MEC  
2. syncs to GitHub as a CSV  
3. is displayed through a custom WordPress plugin using Mapbox  

Analytics are tracked in **GA4** and reported in **Looker Studio**.

This system allows RAM to:

- manage upcoming clinics in WordPress
- display them consistently on the website
- measure how users interact with upcoming clinic information
- report on user interest and engagement over time

