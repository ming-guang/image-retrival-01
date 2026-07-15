#import "./template/lib.typ": declaration-of-independent-processing, report

// Register abbreviations and glossary
#import "dependencies.typ": make-glossary, print-glossary, register-glossary

#let authors = (
  "20127298 - Nguyễn Trần Minh Quang",
)

// Initialize template
#show: report.with(
  language: "en",
  title: "Lab Report - Lab 02",
  author: authors.join("\n"),
  faculty: "Computer Vision",
  department: "Information Technology",
  // Everything inside "before-content" will be automatically injected
  // into the document before the actual content starts.
  before-content: {
    
  },
  // Everything inside "after-content" will be automatically injected
  // into the document after the actual content ends.
  after-content: {
    pagebreak(weak: true)
  },
)

// Include chapters of report
#pagebreak(weak: true)
#include "./chapters/summary.typ"
#pagebreak(weak: true)
#include "./chapters/works.typ"